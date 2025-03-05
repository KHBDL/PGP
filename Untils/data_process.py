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

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: networkx图
            label: 图标签
            node_tags: 节点标签
            node_features: torch float张量，是节点标签的one-hot表示。作为神经网络的输入
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors[i]表示节点i的所有临界点
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

def load_data(dataset, degree_as_tag):
    if dataset == 'ECG':
        x_train = np.load('./dataset/ECG/X_train.npy')
        x_test = np.load('./dataset/ECG/X_test.npy')
        y_train = np.load('./dataset/ECG/y_train.npy',allow_pickle=True)
        y_test = np.load('./dataset/ECG/y_test.npy',allow_pickle=True)
        degree_as_tag = False
        print('loading ECG data')
        g_list = []
        label_dict = {}
        feat_dict = {}
        
        with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
            n_g = int(f.readline().strip())  #图数量
            for i in range(n_g):    
                row = f.readline().strip().split()  #读取每个图对应的行数据
                n, l = [int(w) for w in row] #n: 节点数量； l: 图标签
                if not l in label_dict:  #获取新标签
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g = nx.Graph()
                node_tags = []
                node_features = []
                n_edges = 0
                for j in range(n):
                    g.add_node(j)
                    row = f.readline().strip().split() #遍历某一个图节点对应的邻点
                    tmp = int(row[1]) + 2  
                    if tmp == len(row):   #每一行都代表一个节点，其中第一数字代表节点特征，第二个代表邻点数量，第三个后代表邻点索引
                        # no node attributes
                        row = [int(w) for w in row]  
                        attr = None
                    else:
                        row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    if not row[0] in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[row[0]] = mapped
                    node_tags.append(feat_dict[row[0]])   
        
                    if tmp > len(row):
                        node_features.append(attr) #node attributes作为node features
        
                    n_edges += row[1]
                    for k in range(2, len(row)):
                        g.add_edge(j, row[k])  #添加每一个节点的邻点索引
                        # print(j, ow[k])
        
                if node_features != []:  #有节点特征
                    node_features = np.stack(node_features)
                    node_feature_flag = True
                else:
                    node_features = None  #无节点特征
                    node_feature_flag = False
        
                assert len(g) == n  #确保子图的数量是正确的
                g_list.append(S2VGraph(g, l, node_tags))
        
        #add labels and edge_mat       
        for g in g_list:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)
        
            g.label = label_dict[g.label]
        
            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])
        
            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0,1)
        
        if degree_as_tag:
            for g in g_list:
                g.node_tags = list(dict(g.g.degree).values())
        
        #Extracting unique tag labels   
        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))
        
        tagset = list(tagset)
        tag2index = {tagset[i]:i for i in range(len(tagset))}
        
        k = 0
        for g in g_list:
            if k < len(y_train):
                g.node_features = torch.tensor(x_train[k].T)
            else:
                g.node_features = torch.tensor(x_test[k-len(y_train)].T)
            k+=1
        
        
        print('# classes: %d' % len(label_dict))
        print('# maximum node tag: %d' % len(tagset))
        
        print("# data: %d" % len(g_list))
        return g_list, len(label_dict)
        
    else:
        '''
            dataset: name of dataset
            test_proportion: ratio of test train split
            seed: random seed for random splitting of dataset
        '''
    
        print('loading data')
        g_list = []
        label_dict = {}
        feat_dict = {}
    
        with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
            n_g = int(f.readline().strip())
            for i in range(n_g):
                row = f.readline().strip().split()
                n, l = [int(w) for w in row]
                if not l in label_dict:
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g = nx.Graph()
                node_tags = []
                node_features = []
                n_edges = 0
                for j in range(n):
                    g.add_node(j)
                    row = f.readline().strip().split()
                    tmp = int(row[1]) + 2
                    if tmp == len(row):
                        # no node attributes
                        row = [int(w) for w in row]
                        attr = None
                    else:
                        row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    if not row[0] in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[row[0]] = mapped
                    node_tags.append(feat_dict[row[0]])
    
                    if tmp > len(row):
                        node_features.append(attr)
    
                    n_edges += row[1]
                    for k in range(2, len(row)):
                        g.add_edge(j, row[k])
    
                if node_features != []:
                    node_features = np.stack(node_features)
                    node_feature_flag = True
                else:
                    node_features = None
                    node_feature_flag = False
    
                assert len(g) == n
                
                g_list.append(S2VGraph(g, l, node_tags))
    
        #add labels and edge_mat       
        for g in g_list:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)
    
            g.label = label_dict[g.label]
    
            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])
    
            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0,1)
    
        if degree_as_tag:
            for g in g_list:
                g.node_tags = list(dict(g.g.degree).values())
    
        #Extracting unique tag labels   
        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))
    
        tagset = list(tagset)
        tag2index = {tagset[i]:i for i in range(len(tagset))}
    
        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
    
    
        print('# classes: %d' % len(label_dict))
        print('# maximum node tag: %d' % len(tagset))
    
        print("# data: %d" % len(g_list))
    
        return g_list, len(label_dict)