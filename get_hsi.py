import numpy as np
import tifffile
import scipy.io as sio

def get_data(img_path, label_path):
    if img_path.lower().endswith('.tif'):
        img_data = tifffile.imread(img_path)
        label_data = tifffile.imread(label_path)

        if img_data.ndim == 3 and img_data.shape[0] < img_data.shape[1]:
            # Convert to [H, W, C] format (originally [C, H, W])
            img_data = np.transpose(img_data, [1, 2, 0])

        return img_data.astype('float32'), label_data.astype('int8')

    elif img_path.lower().endswith('.mat'):
        img_mat = sio.loadmat(img_path)
        img_key = [k for k in img_mat.keys() if not k.startswith('__')]

        if label_path is not None:
            gt_mat = sio.loadmat(label_path)
            gt_key = [k for k in gt_mat.keys() if not k.startswith('__')]
            return img_mat[img_key[0]].astype('float32'), gt_mat[gt_key[0]].astype('int8')

        return img_mat[img_key[0]].astype('float32'), img_mat[img_key[1]].astype('int8')

def gaussian_kernel(size, sigma=0.5):
    """
    Generates a size x size Gaussian kernel with standard deviation sigma.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)
