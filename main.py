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
import time
import scipy.sparse as sp
from tqdm import tqdm
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm  
import torch_geometric as tg
from torch_scatter import scatter
from Untils.data_process import load_data,separate_data
from Untils.perturbation import Aggre_Perturb, perturb_adj_continuous
from Untils.transform import index_to_dense,shuffle_and_group
from Untils.calculation import cal_noise_scale
from Untils.model import PGP

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# 固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
        
# 早停法实现
def early_stopping(indicator_style, val_indicator, best_val_indicator, patience, patience_counter, model, model_path):
    """
    基于验证集的损失实现早停法。

    参数：
    - val_indicator: 当前epoch的验证集损失或准确率
    - best_val_indicator: 最佳验证集损失
    - patience: 允许验证集指标没有改进的最大epoch数
    - patience_counter: 当前没有改进的epoch数
    - model: 当前训练的模型
    - model_path: 最佳模型保存路径

    返回：
    - best_val_indicator: 更新后的最佳验证集指标
    - patience_counter: 更新后的没有改进的epoch计数器
    - 是否需要停止训练
    """
    if indicator_style == 'loss':
        if val_indicator < best_val_indicator:
            best_val_indicator = val_indicator
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_path)
            # print('save best model.......')
            return best_val_indicator, patience_counter, False
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                return best_val_indicator, patience_counter, True
            else:
                return best_val_indicator, patience_counter, False
    elif indicator_style == 'acc':
        if val_indicator > best_val_indicator:
            best_val_indicator = val_indicator
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_path)
            return best_val_indicator, patience_counter, False
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                return best_val_indicator, patience_counter, True
            else:
                return best_val_indicator, patience_counter, False

def main():
        # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--device', type=str, default="cuda:0", help="device_name")
    parser.add_argument('--dataset', type=str, default="NCI1", help="dataset")
    parser.add_argument('--level', type=str, default="node", help="dataset")  #edge or node
    parser.add_argument('--merge_mode', type=str, default="ATT", help="merge_mode")  #'MAX', 'NONE', 'ATT'
    
    parser.add_argument('--Fold', type=int, default=0, help="The index of train and test data")
    parser.add_argument('--Hop_1', type=int, default=5, help="The GNN hop of module 1")
    parser.add_argument('--hidden_dimension_2', type=int, default=16, help="The hidden_dimension of module 2")
    parser.add_argument('--Batch_size', type=int, default=32, help="Batch_size")
    parser.add_argument('--epoch_2', type=int, default=50, help="Epoch number of module 2")
    parser.add_argument('--D_max', type=int, default=-1, help="The number of max neighboring nodes") # -1:默认考虑全部邻点

    parser.add_argument('--lr_2', type=float, default=0.01, help="The learning rate of module 2")
    parser.add_argument('--entropy_mean_2', type=float, default=0.001, help="The entropy_mean of module 2")
    parser.add_argument('--Noise_Scale_adj_perturb', type=float, default=2.0, help="The noise scale of adj perturbation")
    
    parser.add_argument('--degree_as_tag', action='store_true', help="The flag of degree as tag")
    parser.add_argument('--Multi_GNN_output_flag_1', action='store_true', help="The flag that module 1 outputs multi GNN results") #一般为True


    # 解析参数
    args = parser.parse_args()
    print("The Parameters：")
    pprint.pprint(vars(args))  # 将 args 对象转换为字典后打印

    device = torch.device(args.device)
    
    dataset = args.dataset   #PROTEINS  #
    degree_as_tag = args.degree_as_tag
    eplison_list = [2,4,8,10,12,16]
    for CC in range(len(eplison_list)):
        eplison_item  = eplison_list[CC]
        seed = 0
        fold_idx = 0
        graphs, num_classes = load_data(dataset, degree_as_tag)
        if dataset == 'ECG':
            train_graphs = graphs[:19634][:int(19634*0.4)] #0.4
            test_graphs = graphs[19634:][:int(2203*0.4)]
        else:
            train_graphs, test_graphs = separate_data(graphs, seed, args.Fold)
        print('The num of Train graph:',len(train_graphs))
        print('The num of Test graph:',len(test_graphs))
                
        # 1. Encoder and Aggre_Perturb  (一次性)
        print('>>>>>>>>>>>>> Step 1. Aggregation & adj Perturbation <<<<<<<<<<<<<<<<<<')
    
        
        Noise_Scale_1 = cal_noise_scale(eplison_item, delta=1e-5, D_max = args.D_max, hop=args.Hop_1, epsilon_e=args.Noise_Scale_adj_perturb, level=args.level)
        Hop_1 = args.Hop_1
        Multi_GNN_output_flag_1 = args.Multi_GNN_output_flag_1
        
        train_aggre_perturb_list = [] 
        adj_noise_list_train = []
        num_classes = 0
        average_nodes = 0
        average_edges = 0
        for index, data in tqdm(enumerate(train_graphs),total=len(train_graphs),leave = True): 
            # 用一次性处理替代循环
            if (data.label+1)>num_classes:
                num_classes = data.label+1
            average_nodes += data.node_features.shape[0]
            average_edges += (data.edge_mat.shape[1])/2
            adj_dense = index_to_dense(data.edge_mat).to(device)
            row_norms = torch.norm(data.node_features, p=2, dim=1, keepdim=True)
            
            # 按行进行 L2 归一化
            x_normalized = data.node_features / row_norms
            feature = torch.tensor(x_normalized).to(device)
            
    
            train_loader_D = NeighborSampler(data.edge_mat, node_idx=None,sizes=[args.D_max], batch_size=feature.shape[0], shuffle=False,num_workers=1) #限制度
            for batch_size, n_id, adjs in train_loader_D:
                adj_dense_D = index_to_dense(adjs.edge_index).to(device)
                Agg_result = Aggre_Perturb(adj_dense_D, feature, noise_scale=Noise_Scale_1, hop=Hop_1, multi_GNN_output_flag = Multi_GNN_output_flag_1)  # Z_0
            
            train_aggre_perturb_list.append(Agg_result)
            adj_noise = perturb_adj_continuous(adj_dense,epsilon= args.Noise_Scale_adj_perturb,noise_type='laplace',noise_seed=46,delta=1e-5).float().to(device)
            adj_noise_list_train.append(adj_noise)
        average_nodes = int(average_nodes/len(train_graphs))
        average_edges = int(average_edges/len(train_graphs))
        print('The average node number is:',average_nodes)
        print('The average edge number is:',average_edges)
        test_aggre_perturb_list = [] 
        adj_list_test = []
        for data in test_graphs:
            adj_dense = index_to_dense(data.edge_mat).to(device)
            row_norms = torch.norm(data.node_features, p=2, dim=1, keepdim=True)
            # 按行进行 L2 归一化
            x_normalized = data.node_features / row_norms
            feature = torch.tensor(x_normalized).to(device)
            Agg_result = Aggre_Perturb(adj_dense, feature, noise_scale=0, hop=Hop_1, multi_GNN_output_flag = Multi_GNN_output_flag_1)  # Z_0
            test_aggre_perturb_list.append(Agg_result)
            adj_list_test.append(adj_dense)
        print('>>>>>>>>>>>>> Step 1: Finished <<<<<<<<<<<<<<<<<<')
        
        ### 2. assignment_matrix_optimation  (需要训练)
        print('>>>>>>>>>>>>> Step 2. PGP Training  <<<<<<<<<<<<<<<<<<')
        num_node_features = train_graphs[0].node_features.shape[-1]
        hidden_dimension_2 = args.hidden_dimension_2  # 16 for protein data
        lr_2 = args.lr_2
        
        if Multi_GNN_output_flag_1:
            model = PGP(input_dim = num_node_features * (Hop_1), hidden_dim = int((hidden_dimension_2 * (Hop_1+1))/2), 
                             output_dim = num_classes, num_clusters = ceil(average_nodes/2), device = device, merge_mode = args.merge_mode).to(device)
        else:
            model = PGP(input_dim = num_node_features, hidden_dim = hidden_dimension_2, 
                             output_dim = num_classes, num_clusters = ceil(average_nodes/2), device = device, merge_mode = args.merge_mode).to(device)   
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_2)  #PROTEIN：0.001
        
        Batch_size = args.Batch_size 
        entropy_mean_2 = args.entropy_mean_2
        epoch_2 = args.epoch_2
        train_graph_index_all = shuffle_and_group(len(train_graphs), Batch_size)
    
        test_acc_folder = []
        for K in range(len(train_graph_index_all)):
            print(f"Training fold {K+1}/{len(train_graph_index_all)}...")
            train_graph_index = train_graph_index_all[:K] + train_graph_index_all[K+1:]
            vaild_graph_index = train_graph_index_all[K]
        
            # 早停参数
            patience = int(epoch_2/2)  # 设置耐心为0.5
            indicator_style = 'loss'
            if indicator_style == 'loss':
                best_val_indicator = float('inf')  # 初始验证损失为正无穷
            elif indicator_style == 'acc':
                best_val_indicator = 0
            patience_counter = 0  # 初始没有达到早停条件
            model_path_fold = f"./model_save/PGP_Graph_Classification/best_model_fold{K}_"+args.dataset+".pth"  # 设置每个fold保存的最佳模型路径
            
            for epoch in range(epoch_2):
                model.train() #交叉训练集
                total_loss = 0
                correct = 0
                x_ass_list_train = []
                s_purturb_list_train = []
                iter_num = 0
                for i in range(len(train_graph_index)):
                    optimizer.zero_grad()
                    out_all = []
                    target_all = []
                    entropy_mean_all = 0
                    for j in range(len(train_graph_index[i])):
                        index_graph = train_graph_index[i][j]
                        x_ass, cluster_assignments, entropy_mean, out, graph_emb = model(train_aggre_perturb_list[index_graph],adj_noise_list_train[index_graph])  
                        x_ass_list_train.append(x_ass)
                        s_purturb_list_train.append(cluster_assignments)
                        out_all.append(out)
                        target_all.append(train_graphs[index_graph].label)
                        entropy_mean_all += entropy_mean
            
            
                    out_all = torch.stack(out_all, dim=0).to(device)  
                    target_all = torch.tensor(target_all).view(-1).to(device)  
                    # 计算损失 
                    loss = criterion(out_all, target_all)  #PROTEIN：0.1
                    loss.backward()
                    optimizer.step()
                    iter_num += 1
                    total_loss += loss.item()
                    pred = out_all.argmax(dim=1)
                    correct += (pred == target_all).sum().item()
                train_loss = total_loss / iter_num
                train_acc = correct / len(train_graphs)
    
                model.eval() #交叉验证集
                correct = 0
                val_loss = 0
                with torch.no_grad():
                    x_ass_list_test = []
                    s_purturb_list_test = []
                    out_all = []
                    target_all = []
                    for k in vaild_graph_index:
                        x_ass, cluster_assignments, entropy_mean, out, graph_emb = model(train_aggre_perturb_list[k],adj_noise_list_train[k])  
                        x_ass_list_test.append(x_ass)
                        s_purturb_list_test.append(cluster_assignments)
                        out_all.append(out)
                        target_all.append(train_graphs[k].label)
                        # 将所有输出拼接起来
                    out_all = torch.stack(out_all, dim=0).to(device)  
                    target_all = torch.tensor(target_all).view(-1).to(device)  
                    loss = criterion(out_all, target_all)
                    val_loss += loss.item()
                    pred = out_all.argmax(dim=1)
                    correct += (pred == target_all).sum().item()
                vaild_acc = correct / len(vaild_graph_index)
                val_loss /= len(vaild_graph_index)
                if indicator_style == 'loss':
                    val_indicator = val_loss
                elif indicator_style == 'acc':
                    val_indicator = vaild_acc
                
    
                best_val_indicator, patience_counter, stop_training = early_stopping(indicator_style, val_indicator, best_val_indicator, patience, patience_counter, model, model_path_fold)
    
                # 如果满足早停条件，则停止训练
                if stop_training:
                    break
                
            
                
            model.eval()  #测试集
            correct = 0
            with torch.no_grad():
                x_ass_list_test = []
                s_purturb_list_test = []
                out_all = []
                target_all = []
                for k in range(len(test_graphs)):
                    x_ass, cluster_assignments, entropy_mean, out, graph_emb = model(test_aggre_perturb_list[k],adj_list_test[k])  
                    x_ass_list_test.append(x_ass)
                    s_purturb_list_test.append(cluster_assignments)
                    out_all.append(out)
                    target_all.append(test_graphs[k].label)
                    # 将所有输出拼接起来
                out_all = torch.stack(out_all, dim=0).to(device)  
                target_all = torch.tensor(target_all).view(-1).to(device)  
                pred = out_all.argmax(dim=1)
                correct += (pred == target_all).sum().item()
                test_acc = correct / len(test_graphs)
        
            test_acc_folder.append(round(test_acc, 4))
        print(f'Epsilon {eplison_item}, Noise_scale {Noise_Scale_1}, The test acc in all {len(train_graph_index_all)}. Cross vaild folds:')
        print(test_acc_folder)
        print('>>>>>>>>>>>>> Step 2: Finished <<<<<<<<<<<<<<<<<<')


if __name__ == "__main__":
    main()