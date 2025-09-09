import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.exposure import rescale_intensity
from skimage.feature import canny

def estimate_superpixel_count(hsi, alpha=3000, sigma=1.0, visualize=False):
    H, W, C = hsi.shape
    N = H * W

    # reshape -> [N, C]
    hsi_reshaped = hsi.reshape(-1, C)

    # PCA -> 第一个主成分
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(hsi_reshaped)  # shape: [N, 1]
    pc1_image = pc1.reshape(H, W)

    # normalize to [0, 1] range
    pc1_norm = rescale_intensity(pc1_image, out_range=(0, 1))

    # edge detection
    edge_map = canny(pc1_norm, sigma=sigma)
    N_edge = np.count_nonzero(edge_map)

    # 可视化边缘图
    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(edge_map, cmap='gray')
        plt.title('Edge Map from PC1')
        plt.axis('off')
        plt.show()

    M = int(alpha * (N_edge / N))
    return max(M, 1)
