import numpy as np
import torch
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_similarity
from evaluation import cluster_accuracy
from visualize import visualize_segmentation_map

def affinity_to_pixellabels(affinity_mat, n_clusters):
    affinity_mat = np.abs(affinity_mat)
    spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize', random_state=42)
    spectral.fit(affinity_mat)
    y_pre = spectral.fit_predict(affinity_mat)
    y_pre = y_pre.astype('int')
    return y_pre

def spixel_to_pixel_labels(sp_level_label, H, select_sp):
    # print("sp_level_label", np.unique(sp_level_label))
    res = np.ones_like(H) * 100
    for i in range(len(sp_level_label)):
        # print(sp_level_label[i])
        # print(np.where(H == select_sp[i]))
        res[np.where(H == select_sp[i])] = sp_level_label[i]
    # print("res", np.unique(res))
    return res
def subspace_clustering(C, labels, H, epoch, logger):
    '''
    Args:
        C: The self-representation of all superpixels
        labels: the label of labeled data, two dimension, so have spatial information
        H: The label that any pixel belongs to any superpixel.
        epoch: current epoch, check it to see if it needs to evaluate

    Returns: the clustering label
    '''
    # step 1, get the representation matrix of target superpixel which in the clustering area,
    # get all representation parameter
    H = torch.squeeze(H)
    H = H.cpu().numpy()

    # 这里的H可能有跳跃的，这里需要重映射一下，匹配超像素
    super_pixel_index = np.unique(H)
    for i, old_index in enumerate(super_pixel_index):
        H[H == old_index] = i


    C = C.detach().cpu().numpy()
    C_reorder = np.sort(C, axis=0)[::-1]


    labels = labels.T
    h, w = labels.shape
    logger.info(f"H: {h}, W: {w}")
    # visualize_segmentation_map(H.reshape([h, w]))
    # visualize_segmentation_map(labels.reshape([h, w]))

    H = (H.reshape([h, w])).reshape(-1)
    labels = (labels.reshape([h, w])).reshape(-1)

    index, num_class = np.unique(labels, return_counts=True)
    indx = np.where(labels > 0)[0]
    H_selected = H[indx]
    select_sp = np.unique(H_selected)

    C = np.abs(C)

    C = C + C.T
    C_selected = C[:, select_sp]
    affinity_mat_ = cosine_similarity(C_selected.T)  # shape (N, N), 值域 [-1, 1]
    number_classes = len(np.unique(labels)) - 1
    y_pre_sp = affinity_to_pixellabels(affinity_mat_, number_classes)
    logger.info(f"{np.unique(y_pre_sp)}")

    y_predict = spixel_to_pixel_labels(y_pre_sp, H, select_sp)

    visualize_segmentation_map(y_predict.reshape([w,h]))
    y_target = labels[indx]
    y_predict = y_predict[indx]

    logger.info(f"y_target: {np.unique(y_target)}")
    logger.info(f"y_predict: {np.unique(y_predict)}")

    y_best, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_true=y_target, y_pre=y_predict, return_aligned=True)

    # logger.info(f"y_target: {np.unique(y_target)}")
    # logger.info(f"y_predict: {np.unique(y_best)}")

    a = np.zeros_like(labels)
    a[labels > 0] = y_best

    visualize_segmentation_map(a.reshape([h, w]))

    logger.info(
        'Epoch = {:} OA = {:.4f} Kappa = {:.4f} NMI = {:.4f}'.format(epoch, acc, kappa, nmi))

    return acc, kappa, nmi