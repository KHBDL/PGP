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


def cal_privacy_budget(Hop_num,Degree_max, sigma_noise, adj_perturb_epsilon, privacy_level, delta=1e-5):
    if privacy_level == 'node':
        Degree_max = np.sqrt(Degree_max**2+ Degree_max)
        epsilon_sum = Hop_num*Degree_max/(2*(sigma_noise**2))+np.sqrt(2*Hop_num*Degree_max*np.log(1/delta))/sigma_noise + adj_perturb_epsilon
    else:
        epsilon_sum = np.sqrt(2)*Hop_num/(2*(sigma_noise**2))+np.sqrt(2*np.sqrt(2)*Hop_num*np.log(1/delta))/sigma_noise + adj_perturb_epsilon
    return epsilon_sum

def cal_noise_scale(epsilon, delta, D_max, hop, epsilon_e, level):
    if level=='node':
        sensitivity = np.sqrt(D_max+D_max**2) 
    elif level=='edge':
        sensitivity = np.sqrt(2)
    epsilon_ap = epsilon - epsilon_e
    assert epsilon_ap > 0, "epsilon_total must be greater than 0"
    sigma = (np.sqrt(2*sensitivity*hop*np.log(1/delta))+np.sqrt(2*sensitivity*hop*np.log(1/delta)+2*epsilon_ap*np.log(1/delta)*hop))/(2*epsilon_ap)
    noise_scale = sigma/sensitivity
    return sigma