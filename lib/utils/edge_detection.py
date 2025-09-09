import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_hsi_edge_map(hsi_cube, method='mean', visualize=False):
    """
    Parameters:
        hsi_cube: numpy array or torch tensor of shape (H, W, C)，无 batch
        method: 'mean' | 'max' | 'sum'  聚合方式
        visualize: bool 是否可视化
    Returns:
        edge_map: numpy array of shape (H, W)
    """

    if isinstance(hsi_cube, np.ndarray):
        hsi_cube = torch.from_numpy(hsi_cube).float()

    H, W, C = hsi_cube.shape  # (H, W, C)
    hsi_cube = hsi_cube.permute(2, 0, 1)  # (C, H, W)

    # Sobel kernel
    sobel_x_kernel = torch.tensor([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]], dtype=torch.float32)
    sobel_y_kernel = torch.tensor([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]], dtype=torch.float32)

    # 扩展到每个通道一个核
    sobel_x = sobel_x_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # (C,1,3,3)
    sobel_y = sobel_y_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # (C,1,3,3)

    sobel_x = sobel_x.to(hsi_cube.device)
    sobel_y = sobel_y.to(hsi_cube.device)

    # 加入 fake batch 维度 (1,C,H,W)，为了用 F.conv2d
    hsi_cube = hsi_cube.unsqueeze(0)  # (1,C,H,W)

    grad_x = F.conv2d(hsi_cube, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(hsi_cube, sobel_y, padding=1, groups=C)

    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # (1, C, H, W)

    # 聚合
    if method == 'mean':
        edge_map = grad_mag.mean(dim=1)  # (1, H, W)
    elif method == 'max':
        edge_map = grad_mag.max(dim=1)[0]
    elif method == 'sum':
        edge_map = grad_mag.sum(dim=1)
    else:
        raise ValueError(f"Unsupported method: {method}")

    edge_map = edge_map.squeeze(0).cpu().numpy()  # (H, W)

    if visualize:
        plt.figure(figsize=(6,6))
        plt.title(f"HSI Edge Map ({method})")
        plt.imshow(edge_map, cmap='gray')
        plt.axis('off')
        plt.show()

    return edge_map
