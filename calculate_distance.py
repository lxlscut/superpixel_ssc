import time

import torch

class Distance():
    def __init__(self, device, init_spix_indices, num_spixels_w, num_spixels_h):
        self.device = torch.device(device)
        self.offset = torch.tensor([
                                        [-1, -1], [0, -1], [1, -1],
                                        [-1, 0], [0, 0], [1, 0],
                                        [-1, 1], [0, 1], [1, 1]
                                        ], device=device)

        self.num_spixels_w = num_spixels_w
        self.num_spixels_h = num_spixels_h
        self.init_spix_indices = init_spix_indices.to(self.device)
        self.x_idx = None
        self.y_idx = None

    def innitialize(self):

        self.x_idx = self.init_spix_indices % self.num_spixels_w
        self.y_idx = self.init_spix_indices // self.num_spixels_w

        # 3. 计算所有偏移后的坐标 [9, N]
        self.x_candidates = self.x_idx.unsqueeze(0) + self.offset[:, 0].unsqueeze(1)  # [9, N]
        self.y_candidates = self.y_idx.unsqueeze(0) + self.offset[:, 1].unsqueeze(1)  # [9, N]

        # 4. 计算合法性掩码 [9, N]
        valid_mask = (self.x_candidates >= 0) & (self.x_candidates < self.num_spixels_w) & \
                     (self.y_candidates >= 0) & (self.y_candidates < self.num_spixels_h)

        # 5. 计算候选超像素索引 [9, N]
        candidate_indices = self.y_candidates * self.num_spixels_w + self.x_candidates  # [9, N]

        # 7. 只处理合法位置（避免无效索引）
        valid_pos = valid_mask.nonzero(as_tuple=True)  # 返回 (行索引, 列索引) 的元组
        self.offset_idx, self.pixel_idx = valid_pos[0], valid_pos[1]

        # 8. 获取所有合法位置的候选超像素索引
        self.sp_idx = candidate_indices[self.offset_idx, self.pixel_idx]# [valid_count]

        self.pixel_feat_valid = None

        # Step 1: 对 pixel_idx 进行排序，得到排序索引
        # sort_idx = self.pixel_idx.argsort()
        #
        # # Step 2: 按照排序索引同步排序 offset_idx 和 sp_idx
        # self.pixel_idx = self.pixel_idx[sort_idx]
        # self.offset_idx = self.offset_idx[sort_idx]
        # self.sp_idx = self.sp_idx[sort_idx]

    @torch.compile(mode="max-autotune")
    def compute(self, pixel_features, spixel_features, weight_feature, weight_spatial):
        C, N = pixel_features.shape
        dtype = pixel_features.dtype
        dist_matrix = torch.full((9, N), float(1e15), dtype=dtype, device=self.device)
        # 9. 收集合法位置上的像素特征和对应超像素特征
        # if self.pixel_feat_valid == None:
        pixel_feat_valid = pixel_features[:, self.pixel_idx]  # [C, valid_count]
        sp_feat_valid = spixel_features[:, self.sp_idx]  # [C, valid_count]

        # 10. 分别计算特征距离和空间距离
        # 特征距离 (前 C-2 通道)
        feat_diff = pixel_feat_valid[:-2] - sp_feat_valid[:-2]  # [C-2, valid_count]
        feat_dist = torch.square(feat_diff).sum(dim=0) / (C - 2)  # [valid_count]

        # 空间距离 (最后 2 通道)
        pos_diff = pixel_feat_valid[-2:] - sp_feat_valid[-2:]  # [2, valid_count]
        pos_dist = torch.square(pos_diff).sum(dim=0) / 2  # [valid_count]

        # 11. 获取权重并计算加权距离
        wf = weight_feature[self.sp_idx]  # [valid_count]
        ws = weight_spatial[self.sp_idx]  # [valid_count]
        total_dist = wf * feat_dist + ws * pos_dist  # [valid_count]

        # 12. 填充距离矩阵
        dist_matrix[self.offset_idx, self.pixel_idx] = total_dist

        return dist_matrix
