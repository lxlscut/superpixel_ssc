import numpy as np
import torch
import torch.nn as nn
from ADMM_Self_representation import unfold_admm
from lib.ssn.ssn import ssn_iter
import torch.nn.functional as F

class SSNModel(nn.Module):
    def __init__(self, nspix, labels, data, n_iter=20, origin_nspixle=None, cal=None, device='cuda'):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        # Learnable lambda parameter for ADMM, initialized to -5
        self.lamda = nn.Parameter(torch.ones(25, device=device)*(-5), requires_grad=True)
        self.a = nn.Parameter(torch.zeros(self.nspix, device=device),requires_grad=True)
        self.noise = nn.Parameter(torch.zeros_like(data, device=device), requires_grad=True)
        self.admm = unfold_admm(nuber_layer=10, initial_rho=1.0, lamda=self.lamda)
        self.temp = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
        self.labels = labels.T
        self.origin_nspixle = origin_nspixle
        self.cal = cal


    def forward(self, Feature, coordinate):
        color_scale = torch.sigmoid(self.a)*0.8+0.1
        pos_scale = 1 - color_scale
        print("color_scale:", torch.mean(color_scale), "pos_scale:", torch.mean(pos_scale), "temp:", self.temp)
        x = torch.cat([Feature, coordinate], 0)
        abs_affinity, hard_labels, spixel_features, affinity_matrix, pixel_features = ssn_iter(x, self.origin_nspixle, self.n_iter, self.noise, color_scale, pos_scale, self.temp, self.cal)

        # If training separately
        # spixel_features = spixel_features.detach()

        # Get the class of the hard-labels
        sp_indicies = torch.unique(hard_labels)
        spixel_features_show = spixel_features.detach().cpu().numpy()
        spixel_features_selected = spixel_features[:, sp_indicies]
        # Self-representation evaluated only with the feature space
        superpixel_feature = spixel_features_selected[:Feature.shape[0], :]


        superpixel_recon_norm, superpixel_norm, c_i, z_t = self.admm(superpixel=superpixel_feature, indices=sp_indicies)
        return abs_affinity, hard_labels, spixel_features, superpixel_recon_norm, c_i, z_t, superpixel_norm, affinity_matrix
