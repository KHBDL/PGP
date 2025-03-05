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
from tqdm import tqdm
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm  
import torch_geometric as tg
from torch_scatter import scatter

def compute_entropy(tensor):
    """
    计算每一行的熵，并对行进行求和或取平均。
    :param tensor: 输入的 logits 或概率分布 (N, C)，N 是样本数，C 是类别数
    :return: 总体熵，按行求平均或求和
    """
    # 对每行进行 softmax 操作，得到每行的概率分布
    probs = F.softmax(tensor, dim=1)
    
    # 计算每行的熵
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # 加上一个小值避免 log(0)
    
    # 对所有行的熵进行求和或求平均
    return entropy.mean()

class Merge_xs(nn.Module):
    def __init__(self, mode, dim, num_levels, device, drop_ratio = 0.5):
        super(Merge_xs, self).__init__()
        self.mode = mode
        self.dim = dim
        self.num_levels = num_levels
        self.drop_ratio = drop_ratio
        self.device = device

        if self.num_levels > 1:
            if self.mode == 'MAX':
                self.out_cat = JumpingKnowledge(mode='max', channels=self.dim, num_layers=self.num_levels)
            elif self.mode == 'LSTM':
                self.out_cat = JumpingKnowledge(mode='lstm', channels=self.dim, num_layers=self.num_levels)
            elif self.mode == 'ATT':
                self.lin_att = nn.Linear(2* self.dim, 1)

    def forward(self, xs):
        score = None

        if self.mode == 'NONE':
            embedding = xs[0]
        elif self.mode == 'MEAN':
            # mean
            embedding = torch.mean(torch.stack(xs), dim=0)
        elif self.mode == 'ATT':
            query = xs[0]
            message = torch.cat(xs[1:], axis=0)
            N = query.shape[0]
            # normalize inputs
            query = F.normalize(query, p=2, dim=-1)
            message = F.normalize(message, p=2, dim=-1)
            num_levels = len(xs)
            score = self.lin_att(torch.cat((message, query.repeat(num_levels - 1, 1)), dim=-1)).squeeze(-1)
            score = F.leaky_relu(score, inplace=False)
            # sparse softmax
            index = torch.LongTensor(
                [list(range(N, N * (num_levels))), list(range(N)) * (num_levels - 1)]
            ).to(self.device)
            score = tg.utils.softmax(score, index[1], num_nodes=N * num_levels)
            # Sample attention coefficients stochastically.
            score = F.dropout(score, p=self.drop_ratio, training=self.training)
            # add weight to message
            message = score.unsqueeze(-1) * message
            # obtain final embedding
            embedding = query + scatter(message, index[1], dim=0, reduce='add')
        else:
            embedding = self.out_cat(xs=xs)

        return embedding, score
        
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # 定义权重矩阵 W
        self.weight = nn.Parameter(torch.randn(in_features, out_features))  # 权重初始化
        # 定义偏置项
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        """
        :param x: 输入特征矩阵 (N, F_in)
        :param adj: 邻接矩阵 (N, N)
        :return: 输出特征矩阵 (N, F_out)
        """
        # 1. 归一化邻接矩阵
        # 计算度矩阵 D
        # print(x.shape)
        # print(adj.shape)
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        degree = adj.sum(dim=1)  # 每一行的度（行和列都是节点）
        degree_inv_sqrt = torch.pow(degree, -0.5)  # D^(-1/2)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0  # 防止除零

        # 计算归一化邻接矩阵
        norm_adj = adj * degree_inv_sqrt.view(-1, 1) * degree_inv_sqrt.view(1, -1)

        # 2. 线性变换: x * W
        x = torch.mm(x, self.weight)  # x -> (N, F_out)

        # 3. 邻域聚合: norm_adj * x
        out = torch.mm(norm_adj, x)  # out = A_hat * x

        # 4. 添加偏置项并应用激活函数
        out = out + self.bias
        out = F.relu(out)

        return out
        
class DiffPoolLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_clusters):
        super(DiffPoolLayer, self).__init__()
        self.gcn = GCNLayer(in_channels, out_channels)
        self.cluster_layer = nn.Linear(out_channels, num_clusters)
        self.dropout = nn.Dropout(0.5, inplace=True)
    
    def forward(self, x, adj_dense):
        # Graph Convolution Layer
        x_gnn = self.gcn(x, adj_dense)
        x_gnn = F.relu(x_gnn)
        # x = self.dropout(x)
        # 聚合矩阵 (Softmax 分配聚类)
        cluster_assignments = F.softmax(self.cluster_layer(x_gnn), dim=-1)
        entropy_mean = compute_entropy(cluster_assignments)

        x_ass = torch.mm(cluster_assignments.T, x)
        adj_ass = torch.mm(torch.mm(cluster_assignments.T, adj_dense),cluster_assignments)
        f_norm_adj = torch.norm(abs(torch.mm(cluster_assignments,cluster_assignments.T)-adj_dense), p='fro')
        return x_gnn, x_ass, adj_ass,cluster_assignments,entropy_mean,f_norm_adj

class DiffPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters, num_classes,device,merge_mode):
        super(DiffPool, self).__init__()      
        # 第一层 DiffPool
        self.pool1 = DiffPoolLayer(in_channels, hidden_channels, ceil(num_clusters/2))
        self.gcn1 = GCNLayer(in_channels, hidden_channels)
        # 第二层 DiffPool
        self.pool2 = DiffPoolLayer(in_channels, hidden_channels, ceil(num_clusters/4))
        
        # 分类头
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.Merge_embedding = Merge_xs(mode = merge_mode, dim = hidden_channels,  device = device, num_levels = 3)
    
    def forward(self, x, adj_dense):
        # 第一层池化
        x_gnn_1, x_ass_1, adj_ass_1,s_1,entropy_mean_1, f_norm_adj_1 = self.pool1(x, adj_dense)
        # 第二层池化
        x_gnn_2, x_ass_2, adj_ass_2,s_2,entropy_mean_2, f_norm_adj_2 = self.pool2(x_ass_1, adj_ass_1)
        # # 聚合：平均每个簇中的特征
        
        x_gnn_1 = F.relu(x_gnn_1)
        x_1 = torch.mean(x_gnn_1, dim=0) 
        x_gnn_2 = F.relu(x_gnn_2)
        x_2 = torch.mean(x_gnn_2, dim=0)       
        x_gnn_3 = self.gcn1(x_ass_2, adj_ass_2)
        x_gnn_3 = F.relu(x_gnn_3)
        x_3 = torch.mean(x_gnn_3, dim=0) 
        x_att_combine, score = self.Merge_embedding([x_1.unsqueeze(0), x_2.unsqueeze(0), x_3.unsqueeze(0)])
        # x = torch.mean(x_ass_1, dim=0)  #*************
        # # 分类头
        out = self.fc(x_att_combine.reshape(-1))
        
        return out, x_att_combine
        
class PGP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_clusters,device,merge_mode):
        super(PGP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5, inplace=True)
        self.cluster_layer = nn.Linear(hidden_dim, num_clusters)
        self.DiffPool_network = DiffPool(hidden_dim,int(hidden_dim/2),ceil(num_clusters/2), output_dim,device,merge_mode)

    def forward(self, x, adj_dense):       
        x_encoder = self.fc1(x)
        x_encoder = self.dropout(x_encoder)
        x_encoder = torch.relu(x_encoder)
        cluster_assignments = F.softmax(self.cluster_layer(x_encoder), dim=-1)
        x_ass = torch.mm(cluster_assignments.T, x_encoder)
        adj_ass = torch.mm(torch.mm(cluster_assignments.T, adj_dense.float()),cluster_assignments)
        
        x_diff_o,x_att_combine = self.DiffPool_network(x_ass,adj_ass)
        x_o = self.fc2(x_ass)
        x_o = torch.relu(x_o)
        entropy_mean = compute_entropy(cluster_assignments)
        x_mean = torch.mean(x_o, dim=0)
        # x = self.fc2(x)
        return x_ass, cluster_assignments, entropy_mean, x_diff_o, x_att_combine

        # x_encoder = torch.relu(self.fc1(x))
        # x_mean = torch.mean(x_encoder, dim=0)
        # x_o = torch.relu(self.fc2(x_mean))
        # cluster_assignments = F.softmax(self.cluster_layer(x_encoder), dim=-1)
        # entropy_mean = compute_entropy(cluster_assignments)
        # # x = self.fc2(x)
        # return x_encoder, cluster_assignments, entropy_mean, x_o