import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,JumpingKnowledge
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from math import ceil
from torch_geometric.utils import to_dense_adj, to_dense_batch,unbatch_edge_index
import numpy as np
import networkx as nx
import random
from sklearn.model_selection import StratifiedKFold
import math
import argparse
import pprint  
import warnings
from scipy.sparse import coo_matrix
import scipy.sparse as sp

def GNN_NAP(adj_matrix_dense, X, noise_scale):
    adj_matrix_dense = adj_matrix_dense + torch.eye(adj_matrix_dense.size(0)).to(adj_matrix_dense.device)
    aggre_result = torch.mm(adj_matrix_dense.T,X.to(torch.float32))
    aggre_result += creat_noise(aggre_result,noise_scale)  #聚合扰动
    norm_vals = aggre_result.norm(p=2, dim=1, keepdim=True)  # 计算每一行的 L2 范数
    aggre_result_nor = aggre_result/norm_vals  
    return aggre_result_nor

def creat_noise(array,noise_scale):
    # 构建一个与二维数组同维度的高斯白噪声矩阵
    # noise = torch.randn_like(array)  # 使用 randn_like 生成与 array 相同维度的标准正态分布噪声
    # # 如果需要可以通过乘以一个常数来调整噪声的标准差（例如，调整噪声强度）
    # std_dev = noise_scale # 设置噪声标准差
    # scaled_noise = noise * std_dev  # 按照设定的标准差缩放噪声
    noise = torch.normal(0, noise_scale, size=array.shape).to(array.device)
    return noise
    
def Aggre_Perturb(adj_dense, x, noise_scale,hop,multi_GNN_output_flag):  
    # Graph Convolution Layer
    if multi_GNN_output_flag: #是否采用GNN多层输出
        out_all = []
        for i in range(hop):
            # if i == 0:  #将原始特征向量也进行堆叠
            #     out_all.append(x)
            x = GNN_NAP(adj_dense, x, noise_scale)
            out_all.append(x)
        stacked_output = torch.cat(out_all, dim=1)  # 沿着维度1堆叠
        return stacked_output
    else:
        for i in range(hop):
            x = GNN_NAP(adj_dense, x, noise_scale)
        return x

def add_noise_to_gradients(model, noise_multiplier, max_grad_norm):
    """
    Adds noise to gradients and performs gradient clipping.

    :param model: The model whose gradients are being modified.
    :param noise_multiplier: The factor by which noise is added to gradients.
    :param max_grad_norm: The maximum allowed gradient norm for clipping.
    """
    # First, apply gradient clipping
    for param in model.parameters():
        if param.grad is not None:
            # Clip gradients to max_grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Now, add noise to the gradients
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_multiplier
            param.grad += noise

def randomized_response(binary_tensor, epsilon):
    """
    应用随机响应机制到二进制张量。

    :param binary_tensor: 二进制张量 (N, D)，其中 N 是样本数，D 是特征维度
    :param epsilon: 隐私参数 (float)
    :return: 扰动后的二进制张量
    决定了扰动的程度。较小的𝜖提供更强的隐私保护，但可能导致数据实用性降低；较大的𝜖提供更高的数据实用性，但隐私保护较弱。
    扰动概率p：与ϵ相关，控制了每个位被真实报告或被翻转的概率。
    """
    # 计算 p 的值
    p = math.exp(epsilon) / (1 + math.exp(epsilon))
    
    # 生成与 binary_tensor 形状相同的随机张量，值在 [0, 1) 之间
    random_tensor = torch.rand_like(binary_tensor, dtype=torch.float)
    
    # 创建一个掩码，决定哪些位需要翻转
    flip_mask = random_tensor > p
    
    # 根据掩码翻转位
    perturbed = torch.where(flip_mask, 1 - binary_tensor, binary_tensor)
    
    return perturbed

def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):  # https://github.com/AI-secure/LinkTeller/blob/master/worker.py
    np.random.seed(seed)
    if noise_type == 'laplace':
        return np.random.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2 * np.log(1.25 / delta))
        return np.random.normal(0, c * sensitivity / eps, size)
    else:
        raise NotImplementedError(f'Noise type {noise_type} not implemented!')

def perturb_adj_continuous(adj_matrix, epsilon=10, noise_type='laplace', noise_seed=46, delta=1e-5):
    adj = coo_matrix(adj_matrix.cpu())
    n_nodes, n_edges = adj.shape[0], len(adj.data) // 2

    # Get the lower triangle of the adjacency matrix
    A = sp.tril(adj, k=-1)
    epsilon_1 = epsilon * 0.6
    epsilon_2 = epsilon - epsilon_1
    # Generate noise for perturbation
    noise = get_noise(noise_type, (n_nodes, n_nodes), noise_seed, eps=epsilon_2, delta=delta)
    noise *= np.tri(*noise.shape, k=-1, dtype=bool)  # Apply noise only to the lower triangle
    A += noise  # Add noise to the matrix

    # Perturb the number of edges
    n_edges_keep = n_edges + int(get_noise(noise_type, (1,), noise_seed, eps=epsilon_1, delta=delta)[0])
    # print(f'edge number from {n_edges} to {n_edges_keep}')
    # Flatten the matrix to work with edges directly
    a_r = A.A.ravel()

    # Partition the matrix and select the top edges
    n_splits = 2
    len_h = len(a_r) // n_splits
    ind_list = []

    for i in range(n_splits):
        ind_range = a_r[len_h * i:len_h * (i + 1)] if i < n_splits - 1 else a_r[len_h * i:]
        ind_list.append(np.argpartition(ind_range, -n_edges_keep)[-n_edges_keep:] + len_h * i)

    # Final indices selection and preparation of data
    ind_subset = np.hstack(ind_list)
    a_subset = a_r[ind_subset]
    ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]
    
    # Convert flattened indices back to row and column indices
    row_idx, col_idx = divmod(ind_subset[ind], n_nodes)

    # Prepare the data for the sparse matrix
    data_idx = np.ones(n_edges_keep, dtype=np.int32)
    mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(n_nodes, n_nodes))

    # Return the symmetric matrix (no need for floating-point values, only 0 and 1)
    return torch.tensor((mat + mat.T).todense())  # Ensuring the matrix is symmetric and dense

    
def Assignment_purturb(s_purturb, adj_dense_purturb, device):
    # adj_dense_purturb = randomized_response(adj_dense, epsilon=0.5)  ###########################epsilon为重要参数
    adj_ass = torch.mm(torch.mm(s_purturb.T, adj_dense_purturb.float()),s_purturb)
    adj_ass = F.softmax(adj_ass, dim=1) 
    # adj_ass += creat_noise(adj_ass,noise_scale)
    # norm_vals = adj_ass.norm(p=2, dim=1, keepdim=True)  # 计算每一行的 L2 范数
    # adj_ass = adj_ass/norm_vals  
    # adj_ass = F.softmax(adj_ass, dim=-1)   # 否则新邻接矩阵会非常奇怪
    return adj_ass