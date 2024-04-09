import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import torch.distributed as dist

import hyptorch.pmath as pmath
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix, poincare_mean

class HIERPROXYLoss(nn.Module):
    def __init__(self, num_clusters, sz_embed, mrg=0.1, tau=0.1, hyp_c=0.1, clip_r=2.3):
        super().__init__()
        self.num_clusters = num_clusters
        self.sz_embed = sz_embed
        self.tau = tau
        self.hyp_c = hyp_c
        self.mrg = mrg
        self.clip_r = clip_r
        
        self.to_hyperbolic = hypnn.ToPoincare(c=hyp_c, ball_dim=sz_embed, riemannian=True, clip_r=clip_r, train_c=False)
                
        if hyp_c > 0:
            self.dist_f = lambda x, y: dist_matrix(x, y, c=hyp_c)
        else:
            self.dist_f = lambda x, y: 1 - F.linear(F.normalize(x, dim=-1), F.normalize(y, dim=-1))
            
    def compute_centers(self, x, psedo_labels):
        num_cluster = self.num_clusters
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(num_cluster, n_samples).to(x)  # L, N
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        
        centers = torch.zeros(num_cluster, self.sz_embed).to(x)
        if self.hyp_c == 0:
            centers = torch.mm(weight, x)
            centers = F.normalize(centers, dim=1)
            return centers
        else:
            for i in range(len(psedo_labels.unique())):
                centers[psedo_labels.unique()[i]] = poincare_mean(x[psedo_labels==psedo_labels.unique()[i], :], c=self.hyp_c)
            return centers
    
    def binarize_and_smooth_labels(self, T, nb_classes, smoothing_const = 0.1):
        # Optional: BNInception uses label smoothing, apply it for retraining also
        # "Rethinking the Inception Architecture for Computer Vision", p. 6
        import sklearn.preprocessing
        T = T.cpu().numpy()
        T = sklearn.preprocessing.label_binarize(
            T, classes = range(0, nb_classes)
        )
        T = T * (1 - smoothing_const)
        T[T == 0] = smoothing_const / (nb_classes - 1)
        T = torch.FloatTensor(T).cuda()
        return T
    
    def forward(self, z_s, labels):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """        
        if self.hyp_c > 0:
            z_s = self.to_hyperbolic(z_s)
        # centers = self.to_hyperbolic(centers) # 这里直接计算centers

        # 新版本
        centers = self.compute_centers(z_s, labels)
        
        all_dist_matrix = self.dist_f(z_s, centers)
        
        labels = self.binarize_and_smooth_labels(labels, self.num_clusters)
          
        loss = torch.sum(-labels * F.log_softmax(-all_dist_matrix, -1), -1)
        
        # loss v2
        # loss = torch.sum(labels*all_dist_matrix, 1)
        return loss.mean(), centers
    
    
class HIERLoss(nn.Module):
    def __init__(self, sz_embed, mrg=0.1, tau=0.1, hyp_c=0.1, clip_r=2.3):
        super().__init__()
        self.sz_embed = sz_embed
        self.tau = tau
        self.hyp_c = hyp_c
        self.mrg = mrg
        self.clip_r = clip_r
        
        self.to_hyperbolic = hypnn.ToPoincare(c=hyp_c, ball_dim=sz_embed, riemannian=True, clip_r=clip_r, train_c=False)
                
        if hyp_c > 0:
            self.dist_f = lambda x, y: dist_matrix(x, y, c=hyp_c)
        else:
            self.dist_f = lambda x, y: 1 - F.linear(F.normalize(x, dim=-1), F.normalize(y, dim=-1))
    
    def compute_gHHC(self, z_s, lcas, dist_matrix, indices_tuple, sim_matrix):
        i, j, k = indices_tuple
        bs = len(z_s)
        
        cp_dist = dist_matrix
        
        max_dists_ij = torch.maximum(cp_dist[i], cp_dist[j])
        lca_ij_prob = F.gumbel_softmax(-max_dists_ij / self.tau, dim=1, hard=True)
        lca_ij_idx = lca_ij_prob.argmax(-1)
        
        max_dists_ijk = torch.maximum(cp_dist[k], max_dists_ij)
        lca_ijk_prob = F.gumbel_softmax(-max_dists_ijk / self.tau, dim=1, hard=True)
        lca_ijk_idx = lca_ijk_prob.argmax(-1)
        
        dist_i_lca_ij, dist_i_lca_ijk = (cp_dist[i] * lca_ij_prob).sum(1), (cp_dist[i] * lca_ijk_prob).sum(1)
        dist_j_lca_ij, dist_j_lca_ijk = (cp_dist[j] * lca_ij_prob).sum(1), (cp_dist[j] * lca_ijk_prob).sum(1)
        dist_k_lca_ij, dist_k_lca_ijk = (cp_dist[k] * lca_ij_prob).sum(1), (cp_dist[k] * lca_ijk_prob).sum(1)
                    
        hc_loss = torch.relu(dist_i_lca_ij - dist_i_lca_ijk + self.mrg) \
                    + torch.relu(dist_j_lca_ij - dist_j_lca_ijk + self.mrg) \
                    + torch.relu(dist_k_lca_ijk - dist_k_lca_ij + self.mrg)
                                        
        hc_loss = hc_loss * (lca_ij_idx!=lca_ijk_idx).float()
        loss = hc_loss.mean()
                
        return loss
        
    def get_reciprocal_triplets(self, sim_matrix, topk=20, t_per_anchor = 100):
        anchor_idx, positive_idx, negative_idx = [], [], []
        
        topk_index = torch.topk(sim_matrix, topk)[1]
        nn_matrix = torch.zeros_like(sim_matrix).scatter_(1, topk_index, torch.ones_like(sim_matrix))
        sim_matrix = ((nn_matrix + nn_matrix.t())/2).float()         
        sim_matrix = sim_matrix.fill_diagonal_(-1)
                
        for i in range(len(sim_matrix)):
            if len(torch.nonzero(sim_matrix[i]==1)) <= 1:
                continue
            pair_idxs1 = np.random.choice(torch.nonzero(sim_matrix[i]==1).squeeze().cpu().numpy(), t_per_anchor, replace=True)
            pair_idxs2 = np.random.choice(torch.nonzero(sim_matrix[i]<1).squeeze().cpu().numpy(), t_per_anchor, replace=True)              
            positive_idx.append(pair_idxs1)
            negative_idx.append(pair_idxs2)
            anchor_idx.append(np.ones(t_per_anchor) * i)
        anchor_idx = np.concatenate(anchor_idx)
        positive_idx = np.concatenate(positive_idx)
        negative_idx = np.concatenate(negative_idx)
        return anchor_idx, positive_idx, negative_idx
    
    def forward(self, z_s, y, topk=30):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """        
        bs = len(z_s)
        dist_matrix = self.dist_f(z_s, z_s)
          
        sim_matrix = torch.exp(-dist_matrix).detach()
        one_hot_mat = (y.unsqueeze(1) == y.unsqueeze(0))
        sim_matrix[one_hot_mat] += 1
        
        indices_tuple = self.get_reciprocal_triplets(sim_matrix, topk=topk, t_per_anchor = 50)
        loss = self.compute_gHHC(z_s, z_s, dist_matrix, indices_tuple, sim_matrix)
        
        return loss
