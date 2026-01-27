# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import mean_absolute_error
import torch.nn as nn 
 
# define peak area in ground truth data
# 计算峰值误差。将真实值低于阈值的部分置0，同时对预测值进行相同位置的置0处理，然后计算MAE
# 这专门用于评估峰值（流行病高峰）预测性能。
def peak_error(y_true_states, y_pred_states, threshold): 
    # masked some low values (using training mean by states)
    y_true_states[y_true_states < threshold] = 0
    mask_idx = np.argwhere(y_true_states <= threshold)
    for idx in mask_idx:
        y_pred_states[idx[0]][idx[1]] = 0
    # print(y_pred_states,np.count_nonzero(y_pred_states),np.count_nonzero(y_true_states))
    
    peak_mae_raw = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    peak_mae = np.mean(peak_mae_raw)
    # peak_mae_std = np.std(peak_mae_raw)
    return peak_mae
   
def normalize_adj2(adj):
    """Symmetrically normalize adjacency matrix."""
    # 对称归一化邻接矩阵。使用D^{-1/2}AD^{-1/2}公式，这是图卷积网络（GCN）常用的归一化方式。
    # 类似于data.py中load_sim_mat方法得到的self.adj

    # adj += sp.eye(adj.shape[0])   #没有self-loops时通过这条代码添加self-loops
    
    # 1. 转换为 COO 稀疏矩阵  原因：图邻接矩阵非常稀疏（大部分节点不相连）, COO 格式节省内存，构造简单
    adj = sp.coo_matrix(adj)  

    rowsum = np.array(adj.sum(1))   
    # adj.sum(1)：返回形状为 (n, 1) 的稀疏矩阵    scipy的稀疏矩阵sum方法返回的仍然是稀疏矩阵
    # np.array(...)：将稀疏矩阵转换为密集数组，形状仍然是 (n, 1)   因为下面的np.power() 需要数组输入

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  #.flatten()：将 (n, 1) 变为 (n,)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 对角矩阵也很稀疏，用稀疏格式节省内存，而且后续要与稀疏矩阵做乘法
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
""" 
def normalize(mx):
    #  Row-normalize sparse matrix  (normalize feature)   
    #  可以被import torch.nn.functional as F 中的normalize替代
    rowsum = np.array(mx.sum(1))
    rowsum = np.maximum(rowsum, 1e-12)
    r_inv = np.float_power(rowsum, -1).flatten()
    #r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
"""

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # 将scipy稀疏矩阵转换为PyTorch稀疏张量。这是为了在PyTorch中使用稀疏矩阵运算。
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col)==0:
        print(sparse_mx.row,sparse_mx.col)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

def getLaplaceMat(batch_size, m, adj):
    """
    计算随机游走归一化的拉普拉斯矩阵: L = D^(-1) * A
    
    Args:
        batch_size: 批量大小
        m: 节点数量
        adj: 邻接矩阵 (batch, m, m)，可以是加权或二值化的
    
    Returns:
        laplace_mat: 随机游走归一化的拉普拉斯矩阵 (batch, m, m)
    """
    
    # ==============================================================
    # 1. 创建单位矩阵（用于后续操作）
    # ==============================================================
    # 创建 m×m 的单位矩阵，并移到与adj相同的设备（GPU/CPU）
    identity = torch.eye(m, device=adj.device)  # 形状: (m, m)
    # 增加batch维度并扩展到所有batch
    identity = identity.unsqueeze(0).expand(batch_size, m, m)  # 形状: (batch, m, m)
    
    # ==============================================================
    # 2. 可选：添加自环（GCN的标准做法）
    # ==============================================================
    # 注意：原代码没有添加自环，保持原逻辑
    # 如果想要添加自环，取消下面这行注释：
    # adj = adj + identity  # A' = A + I
    
    # ==============================================================
    # 3. 可选：邻接矩阵二值化
    # ==============================================================
    # 原代码将adj>0设为1，保持这个逻辑
    # 创建全1矩阵用于二值化
    ones_mat = torch.ones(m, device=adj.device)  # 形状: (m,)
    ones_mat = ones_mat.unsqueeze(0).expand(batch_size, m, m)  # 形状: (batch, m, m)
    
    # 二值化：adj>0的位置设为1，否则保持原值
    adj = torch.where(adj > 0, ones_mat, adj)
    
    # ==============================================================
    # 4. 计算度矩阵 D（出度矩阵）
    # ==============================================================
    # 计算每个节点的出度：对每一行求和
    # adj形状: (batch, m, m)，dim=2表示对每行求和
    degree = torch.sum(adj, dim=2)  # 形状: (batch, m)
    
    # 增加一个维度，方便后续操作
    degree = degree.unsqueeze(-1)  # 形状: (batch, m, 1)
    
    # ==============================================================
    # 5. 避免除以零，计算度矩阵的逆 D^(-1)
    # ==============================================================
    # 添加一个极小值，防止除零错误
    degree = degree + 1e-12
    
    # 计算度矩阵的逆: D^(-1)
    # 每个节点的度取倒数: 1/度
    degree_inv = torch.pow(degree, -1)  # 形状: (batch, m, 1)
    
    # ==============================================================
    # 6. 扩展度矩阵的逆为对角矩阵
    # ==============================================================
    # 将 (batch, m, 1) 扩展为 (batch, m, m)
    degree_inv = degree_inv.expand(batch_size, m, m)  # 每列都相同
    
    # 使用单位矩阵确保只有对角线有值
    # 即: D^(-1)_diag = I ⊙ degree_inv，其中⊙是逐元素乘法
    degree_inv_diag = identity * degree_inv  # 形状: (batch, m, m)
    # 结果是对角矩阵，对角线元素是 1/度，非对角线为0
    
    # ==============================================================
    # 7. 计算随机游走归一化的拉普拉斯矩阵: L = D^(-1) * A
    # ==============================================================
    # 矩阵乘法: (batch, m, m) @ (batch, m, m) → (batch, m, m)
    laplace_mat = torch.bmm(degree_inv_diag, adj)
    
    # ==============================================================
    # 8. 返回结果
    # ==============================================================
    return laplace_mat



class RegionBatchNorm1d(nn.Module):
    """
    Region-wise BatchNorm for spatio-temporal models.

    Input shape:
        (B, N, C, L)

    Behavior:
        - BN statistics computed independently per region
        - statistics over (B × L)
        - region-specific affine parameters (gamma / beta)
        - fully parallel, no region loop
    """
    def __init__(self, num_regions, num_channels, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_regions = num_regions
        self.num_channels = num_channels

        # BN without affine (only statistics)
        self.bn = nn.BatchNorm1d(
            num_channels,
            eps=eps,
            momentum=momentum,
            affine=False
        )

        # region-wise affine parameters
        self.weight = nn.Parameter(
            torch.ones(num_regions, num_channels)
        )
        self.bias = nn.Parameter(
            torch.zeros(num_regions, num_channels)
        )

    def forward(self, x):
        """
        x: (B, N, C, L)
        """
        B, N, C, L = x.shape
        assert N == self.num_regions
        assert C == self.num_channels

        # (B, N, C, L) -> (N*B, C, L)
        x_bn = x.permute(1, 0, 2, 3).contiguous()
        x_bn = x_bn.view(N * B, C, L)

        # BN statistics over (B × L), region-independent
        x_bn = self.bn(x_bn)

        # reshape back
        x_bn = x_bn.view(N, B, C, L).permute(1, 0, 2, 3)

        # region-wise affine
        x_bn = (
            x_bn
            * self.weight.unsqueeze(0).unsqueeze(-1)
            + self.bias.unsqueeze(0).unsqueeze(-1)
        )

        return x_bn


class AdvancedTimeConv(nn.Module):
    def __init__(self, args, data_loader):
        super().__init__()
        self.m = data_loader.m
        self.k = args.k
        self.window = args.window
        
        # 多分支卷积
        self.branches = nn.ModuleList([
            self._create_branch(1, self.k, kernel_size=3, dilation=1),
            self._create_branch(1, self.k, kernel_size=5, dilation=2),
            self._create_branch(1, self.k, kernel_size=args.window//2, dilation=4),
            self._create_branch(1, self.k, kernel_size=args.window, dilation=1),
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(len(self.branches) * self.k, self.k * 2),
            nn.BatchNorm1d(self.k * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
    
    def _create_branch(self, in_ch, out_ch, kernel_size, dilation):
        """创建卷积分支"""
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding='same'),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
    
    def forward(self, x):
        """向量化版本，提高性能"""
        # x: (batch, window, m)
        batch_size, window_len, num_regions = x.size()
        
        # 重塑为: (batch*m, 1, window)
        x_reshaped = x.permute(0, 2, 1).contiguous()  # (batch, m, window)
        x_reshaped = x_reshaped.view(-1, window_len)  # (batch*m, window)
        x_reshaped = x_reshaped.unsqueeze(1)  # (batch*m, 1, window)
        
        # 多分支特征提取（向量化）
        branch_features = []
        for branch in self.branches:
            feat = branch(x_reshaped)  # (batch*m, k, 1)
            feat = feat.squeeze(-1)    # (batch*m, k)
            branch_features.append(feat)
        
        # 拼接所有分支特征
        combined = torch.cat(branch_features, dim=-1)  # (batch*m, 4k)
        
        # 特征融合
        combined = self.fusion(combined)  # (batch*m, 2k)
        
        # 恢复形状: (batch, m, 2k)
        h_S_C = combined.view(batch_size, num_regions, -1)
        
        return h_S_C