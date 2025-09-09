import math
import os
import time

import torch

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
import torch.nn.functional as F
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    """
    Calculate initial superpixels.

    Args:
        images: torch.Tensor
            A Tensor of shape (B, C, H, W)
        spixels_width: int
            Initial superpixel width
        spixels_height: int
            Initial superpixel height

    Return:
        centroids: torch.Tensor
            A Tensor of shape (B, C, H * W)
        init_label_map: torch.Tensor
            A Tensor of shape (B, H * W)
        num_spixels_width: int
            Number of superpixels in each column
        num_spixels_height: int
            Number of superpixels in each row
    """
    channels, height, width = images.shape
    device = images.device

    centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))

    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")

    init_label_map = init_label_map.reshape(-1)
    centroids = centroids.reshape(channels, -1)

    return centroids, init_label_map


def get_number_of_superpixels(pixel_features, num_spixels):
    height, width = pixel_features.shape[-2:]
    num_spixels_width = int(math.sqrt(num_spixels * width / height))
    num_spixels_height = int(math.sqrt(num_spixels * height / width))
    spixel_features, init_label_map = \
        calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    return spixel_features.shape[1], num_spixels_width, num_spixels_height, init_label_map.flatten().long()


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    """
    Args:
        init_label_map:
        num_spixels_width:

    Returns:
        Returns the superpixel index along with the 9 superpixel indices around them,
        so they can choose the corresponding superpixel when calculating the distance matrix.
    """
    # init_label_map: shape [n_pixel]
    n_pixel = init_label_map.shape[0]
    device = init_label_map.device

    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)  # shape [9]

    abs_pix_indices = torch.arange(n_pixel, device=device).repeat(9)  # shape [9 * n_pixel]
    abs_spix_indices = (init_label_map[None, :] + relative_spix_indices[:, None]).reshape(-1).long()  # shape [9 * n_pixel]

    return torch.stack([abs_spix_indices, abs_pix_indices], 0)  # shape [2, 9 * n_pixel]


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
    # relative_label = find_sparse_column_max_indices(affinity_matrix)
    relative_label = affinity_matrix.max(dim=0)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.long()


def masked_mean_std(dist_matrix, valid_mask, dim=0, keepdim=True):
    # Calculate the sum after setting invalid values to zero, then divide by the number of valid elements to get the mean.
    valid_float = valid_mask.float()
    sum_valid = (dist_matrix * valid_float).sum(dim=dim, keepdim=keepdim)
    count_valid = valid_float.sum(dim=dim, keepdim=keepdim) + 1e-6
    mean = sum_valid / count_valid

    # Calculate the mean squared difference: only for valid parts.
    diff = (dist_matrix - mean) * valid_float
    var = (diff ** 2).sum(dim=dim, keepdim=keepdim) / count_valid
    std = torch.sqrt(var) + 1e-6
    return mean, std


def ssn_iter(pixel_features, num_spixels, n_iter, noise, weight_feature, weight_spatial, temp, cal):
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6

    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """

    height, width = pixel_features.shape[-2:]
    num_spixels_width = int(math.sqrt(num_spixels * width / height))
    num_spixels_height = int(math.sqrt(num_spixels * height / width))
    spixel_features, init_label_map = \
        calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)

    pixel_features = pixel_features.reshape([pixel_features.shape[0], -1])
    permuted_pixel_features = pixel_features.permute(1, 0).contiguous()
    # init_spix_indices = init_label_map.flatten().long()

    noise = noise.reshape([noise.shape[0], -1])
    pixel_features[:-2,:] = pixel_features[:-2,:] + noise

    permuted_pixel_features[:, :-2] = permuted_pixel_features[:, :-2] + noise.T
    mask = (abs_indices[0] >= 0) * (abs_indices[0] < num_spixels).detach()

    for c_iter in range(n_iter):
        # start = time.time()
        dist_matrix = cal.compute(pixel_features, spixel_features, weight_feature, weight_spatial)
        # end = time.time()
        # print("dist_matrix_time:", end - start)

        affinity_matrix = F.softmax(-dist_matrix/0.1, dim=0)

        reshaped_affinity_matrix = affinity_matrix.reshape(-1)
        # start = time.time()
        sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])
        affinity_dense = sparse_abs_affinity.coalesce().to_dense().contiguous()  # [num_superpixels, num_pixels]
        # Dense matrix multiplication
        spixel_features = torch.matmul(affinity_dense, permuted_pixel_features)  # [num_superpixels, feature_dim]
        # Row normalization
        row_sum = affinity_dense.sum(dim=1, keepdim=True)  # [num_superpixels, 1]
        spixel_features = spixel_features / (row_sum + 1e-16)
        # Transpose back
        spixel_features = spixel_features.permute(1, 0)  # [feature_dim, num_superpixels]
        # end = time.time()
        # print("spixel_features_time:", end - start)

    hard_labels = get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)

    return affinity_dense, hard_labels, spixel_features, affinity_matrix, pixel_features
