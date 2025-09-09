import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import watershed


def reconstruction(assignment: torch.Tensor,
                   labels: torch.Tensor,
                   hard_assignment: torch.Tensor = None) -> torch.Tensor:
    """
    可微分的超像素重构函数，支持稀疏矩阵和反向传播
    Args:
        assignment (torch.Tensor): 超像素分配矩阵，可以是稀疏或密集张量 [n_spixels, n_pixels]
        labels (torch.Tensor): 标签张量 [n_channels, n_pixels]
        hard_assignment (Optional[torch.Tensor]): 硬分配索引 [n_pixels]，可选

    Returns:
        reconstructed_labels (torch.Tensor): 重构后的标签 [n_channels, n_pixels]
    """
    # 确保 labels 是 [n_channels, n_pixels] 形状
    labels = labels.to(assignment.dtype)

    # 处理稀疏和密集矩阵
    if assignment.is_sparse:
        # 计算超像素均值（稀疏版本）
        spixel_sum = torch.sparse.mm(assignment, labels.t())
        spixel_count = torch.sparse.sum(assignment, dim=1).to_dense().unsqueeze(1)
        spixel_mean = spixel_sum / (spixel_count + 1e-16)
    else:
        # 密集矩阵版本
        spixel_sum = assignment @ labels.t()
        spixel_count = assignment.sum(dim=1, keepdim=True)
        spixel_mean = spixel_sum / (spixel_count + 1e-16)

    if hard_assignment is None:
        # 软分配重构
        if assignment.is_sparse:
            reconstructed = torch.sparse.mm(assignment.t(), spixel_mean).t()
        else:
            reconstructed = (assignment.t() @ spixel_mean).t()
    else:
        # 硬分配重构
        reconstructed = spixel_mean[hard_assignment, :].t()

    return reconstructed

def reconstruct_loss_with_mse(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with mse

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment)
    return torch.nn.functional.mse_loss(reconstracted_labels, labels)


def spectral_compact(assignment, Feature, hard_assignment=None):
    '''
    Args:
        Q: Tensor of shape (B, n_spixels, n_pixels)
        Feature: Tensor of shape (B, Feature_dim, n_pixels)
        H: Tensor of shape (B, n_pixels)

    Returns:
        The similarity compact with a superpixel
    '''
    reconstracted_labels = reconstruction(assignment, Feature, hard_assignment=hard_assignment)
    return torch.nn.functional.mse_loss(reconstracted_labels, Feature, reduction='mean')

def remap_labels(labels):
    """
    将标签重新映射为连续的整数，从 0 开始。
    参数:
      labels: numpy array, 2D 标签图
    返回:
      remapped: numpy array, 标签重新映射后的结果
    """
    unique = np.unique(labels)
    remapped = np.zeros_like(labels)
    for new_label, u in enumerate(unique):
        remapped[labels == u] = new_label
    return remapped




import torch
import torch.nn.functional as F

def soft_connectivity_loss(soft_assignments):
    """
    soft_assignments: (W, H, K)
    """

    # 正确 permute，不 reshape！
    soft_assignments = soft_assignments.permute(1, 2, 0)  # (H, W, K)
    H, W, K = soft_assignments.shape

    # Step 1: pad with constant 0
    pad_assignments = F.pad(soft_assignments.permute(2, 0, 1), (1, 1, 1, 1), mode='constant', value=0)  # (K, H+2, W+2)

    # Step 2: fetch and calculate neighbor connections one by one
    center = soft_assignments  # (H, W, K)
    conn_list = []

    for dx, dy in [ (-1, 0), ( 0, -1),( 0, 1),( 1, 0)]:
        neighbor = pad_assignments[:, 1+dy:H+1+dy, 1+dx:W+1+dx].permute(1, 2, 0)  # (H, W, K)
        conn = (center * neighbor).sum(dim=-1)  # (H, W)
        conn_list.append(conn)

    # Step 3: Stack conn results and take maximum
    conn_all = torch.stack(conn_list, dim=0)  # (8, H, W)
    conn_sum = conn_all.sum(dim=0)  # (H, W)

    # Step 4: loss
    loss = (4.0 - conn_sum).mean()

    return loss




import torch

def entropy_loss(probs):
    """
    probs: (N, 9) probability distribution
    returns: scalar loss (mean entropy over all samples)
    """
    # probs: (N, 9)
    probs = torch.permute(probs,[1,0])
    entropy_per_sample = -(probs * (probs + 1e-9).log()).sum(dim=-1)  # (N,)
    loss = entropy_per_sample.mean()  # scalar
    return loss




def compute_sparsity_metrics(C: torch.Tensor, threshold: float = 1e-3, eps: float = 1e-8):
    C_abs = C.abs()

    # 非零率
    nonzero_mask = (C_abs > threshold).float()
    nonzero_ratio_per_col = nonzero_mask.mean(dim=0)
    avg_nonzero_ratio = nonzero_ratio_per_col.mean().item()

    # 熵
    col_sum = C_abs.sum(dim=0, keepdim=True) + eps
    probs = C_abs / col_sum
    entropy_per_col = - (probs * (probs + eps).log()).sum(dim=0)
    avg_entropy = entropy_per_col.mean()

    return avg_nonzero_ratio, avg_entropy


def compute_smooth_nonzero_loss(Z, threshold=1e-4, tau=1e-3, target=0.1):
    soft_indicator = torch.sigmoid((Z.abs() - threshold) / tau)
    soft_nonzero_ratio = soft_indicator.mean()
    return (soft_nonzero_ratio - target) ** 2

