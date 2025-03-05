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


def index_to_dense(index_tensor):
    """
    将邻接矩阵的索引形式转换为稠密格式。

    参数：
    - index_tensor: torch.Tensor，邻接矩阵的索引形式，形状为 (2, N)，其中每列是 (i, j) 表示有边的索引。

    返回：
    - torch.Tensor，稠密矩阵。
    """
    # 根据索引计算稠密矩阵的大小
    size = (index_tensor.max() + 1).item()

    # 创建稠密矩阵并初始化为0
    dense_matrix = torch.zeros((size, size), dtype=torch.int32)

    # 将索引中的值设置为1
    dense_matrix[index_tensor[0], index_tensor[1]] = 1

    return dense_matrix


            
def shuffle_and_group(num_range, k):
    """
    打乱 1-num_range 的列表，并划分为每组 K 个元素的小组。

    参数：
    - num_range: int，表示范围的上限（包含）。
    - k: int，每组的元素数量。

    返回：
    - list，包含多个组的列表，每个组是一个子列表。
    """
    # 创建范围列表
    numbers = list(range(0, num_range))

    # 打乱列表
    random.shuffle(numbers)

    # 按 K 个元素分组
    groups = [numbers[i:i + k] for i in range(0, len(numbers), k)]

    return groups
