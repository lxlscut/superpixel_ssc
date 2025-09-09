import numpy as np
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn.functional as F
import torch.nn as nn
torch.use_deterministic_algorithms(True)


class C_update(nn.Module):
    def __init__(self):
        super(C_update, self).__init__()

    def forward(self, input):
        X, u, Z, rho, W, B = input
        C = torch.matmul(W, X) - torch.matmul(B, u - rho * Z)
        return C

class Z_update(nn.Module):
    def __init__(self):
        super(Z_update, self).__init__()

    def forward(self, input):
        """
        :param input:
        :return: z_s: the sparse value that must reserve
                z_m: the margin value that increase robustness and reduce instability
        """
        rho, C, u = input
        Z = C + u / rho
        Z = Z - torch.diag_embed(torch.diag(Z))
        return Z

class u_update(nn.Module):
    def __init__(self):
        super(u_update, self).__init__()

    def forward(self, input):
        u, rho, C, Z = input
        u = u + rho * (C - Z)
        return u

class AdaptiveSoftshrink(nn.Module):
    def __init__(self, lamda):
        super(AdaptiveSoftshrink, self).__init__()
        self.relu = nn.ReLU()
        self.lamda = lamda

    def forward(self, x, rho, indices, thres):
        # 计算绝对值减去lambda后的结果，并应用ReLU
        shrunk = self.relu(torch.abs(x[indices]) - F.softplus(thres) / rho)
        # 恢复原始值的符号
        z_s = shrunk * torch.sign(x)
        return z_s


class unfold_admm(nn.Module):
    def __init__(self, nuber_layer, initial_rho, lamda):
        super(unfold_admm, self).__init__()
        self.num_layer = nuber_layer
        self.rho = nn.Parameter(torch.tensor(initial_rho, dtype=torch.float32), requires_grad=False)
        self.C_update = C_update()
        self.Z_update = Z_update()
        self.u_update = u_update()
        self.lamda = lamda
        self.softshrink = AdaptiveSoftshrink(self.lamda)


    def updation(self, H, u_0, z_0, W_c, B_c, indices):
        u = u_0
        Z = z_0
        X = F.normalize(H, p=2, dim=0)
        result = []
        for i in range(self.num_layer):
            c_i = self.C_update([X, u, Z, self.rho, W_c, B_c])
            z_i = self.Z_update([self.rho, c_i, u])
            z_t = self.softshrink(z_i, self.rho, indices, self.lamda[i])
            u_i = self.u_update([u, self.rho, c_i, z_t])
            u = u_i
            Z = z_t
            if i == self.num_layer-1:
                y_i = torch.matmul(X, c_i)
                result.append(c_i)
                result.append(y_i)
                result.append(z_t)
                result.append(u_i)
                result.append(X)
            
        return result

    #     return W_c, B_c
    def innitialize_c(self, X, rho):
        # 1. 将输入 X 转换为 64 位浮点数
        X_64 = X.to(torch.float64)
        rho_64 = torch.tensor(rho, dtype=torch.float64, device=X.device) # 确保 rho 也是 64 位

        # 2. 创建 64 位单位矩阵
        I_64 = torch.eye(X_64.size(1), device=X_64.device, dtype=torch.float64)

        # 3. 所有中间计算都在 64 位下进行
        m_64 = 2 * X_64.T @ X_64 + rho_64 * I_64

        # 求解 m_64 @ W_c_64 = 2 * X_64.T
        W_c_64 = torch.linalg.solve(m_64, 2 * X_64.T)

        # 求解 m_64 @ B_c_64 = I_64
        B_c_64 = torch.linalg.solve(m_64, I_64)

        # 4. 将最终结果转回 32 位浮点数
        W_c = W_c_64.to(torch.float32)
        B_c = B_c_64.to(torch.float32)

        return W_c, B_c


    def unfolding_admm(self, H, z_0, indices):
        # initialize variable
        u_0 = torch.ones((H.shape[1], H.shape[1]), device=self.rho.device) * 1e-3
        # initialize W, B
        H_n = F.normalize(input=H, p=2, dim=0)
        W_c, B_c = self.innitialize_c(H_n, self.rho)
        c_i, h_r, z_t, u_i, h = self.updation(H=H_n, u_0=u_0, z_0=z_0, W_c=W_c, B_c=B_c, indices = indices)
        return c_i, h_r, z_t, u_i, h


    def forward(self, superpixel, indices):
        """
        Args:
            superpixel: The input superpixel feature
        Returns:
        """
        z0 = torch.zeros((superpixel.shape[1], superpixel.shape[1]), device=self.rho.device)
        [c_i, superpixel_recon, z_t, u_i, superpixel] = self.unfolding_admm(superpixel, z0, indices)
        return superpixel_recon, superpixel, c_i, z_t
