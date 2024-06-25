import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import Counter
import pickle
import json
import os
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import time
from sklearn.metrics import roc_auc_score
import json
import copy
import os
import numpy as np
import torch
import pyod
import random
from mymodel import *

# 讀csv，回傳data frame
def DataLoader(file_name):
    df=pd.read_csv(file_name)
    print(df.shape)
    print(df.columns)
    return df
# 10:1的資料
def data_sampling(df):
    # print(df)
    fraud_data=df[df['Class']==1]
    normal_data=df[df['Class']==0].sample(n=len(fraud_data)*10,random_state=42)
    # test: normal:492*2=984, fruad:int(492*0.2)=98 
    selected_df_eval=pd.concat([normal_data[:int(len(normal_data)*0.2)],fraud_data[:int(len(fraud_data)*0.2)]])
    # train: normal:492*8=3936, fruad:492-int(492*0.8)= 394
    selected_df_train=pd.concat([normal_data[int(len(normal_data)*0.2):],fraud_data[int(len(fraud_data)*0.2):]])
    
    selected_df_train.reset_index(drop=True,inplace=True)
    selected_df_eval.reset_index(drop=True,inplace=True)
    
    return selected_df_train, selected_df_eval

# 標準化
def data_standardization(df):
    # 假設 df 是你的 DataFrame，columns 是要標準化的欄位名稱列表
    columns_to_normalize = ['Time', 'Amount']  # 填入要標準化的欄位名稱
    # 初始化 StandardScaler 物件
    scaler = StandardScaler()
    # 對指定列進行標準化
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def calculate_cosine_similarity(df, type):
    features=df.drop("Class",axis=1)
    cos_sim=cosine_similarity(features)
    np.save("cos_sim_"+type+".npy",cos_sim)
    print("Cosine similarity calculation completed and saved.") 

def cos_sim_describe(cos_sim):
    # 計算每一列的統計量
    max_values = np.max(cos_sim, axis=1)
    min_values = np.min(cos_sim, axis=1)
    median_values = np.median(cos_sim, axis=1)
    mean_values = np.mean(cos_sim, axis=1)
    std_values = np.std(cos_sim, axis=1)

    # 對這些統計量再次計算統計量
    stats_max = np.max(max_values)
    stats_min = np.min(min_values)
    stats_median = np.median(median_values)
    stats_mean = np.mean(mean_values)
    stats_std = np.std(std_values)

    # 印出結果
    print("Statistics of row statistics:")
    print("  Max:", stats_max)
    print("  Min:", stats_min)
    print("  Median:", stats_median)
    print("  Mean:", stats_mean)
    print("  Std:", stats_std)
    cos_sim_dict={'max':stats_max,'min':stats_min,'median':stats_median
                  ,'mean':stats_mean,'std':stats_std}
    return cos_sim_dict

def graph_construction(cos_sim,df,threshold):
    G=nx.Graph()
    for i in tqdm(range(cos_sim.shape[0]),desc="Add nodes into G..."):
        node_name=i
        feature=(df.iloc[i]).to_dict()
        G.add_node(node_name,**feature)

    for i in tqdm(range(cos_sim.shape[0]),desc="Graph Construction..."):
        for j in range(i+1,cos_sim.shape[1]):
            if cos_sim[i,j]>threshold:
                G.add_edge(i,j)
    return G


def graph_describe(G,round_digit):
    density=2*G.number_of_edges()/(G.number_of_nodes()*(G.number_of_nodes()-1))
    # 获取节点的度数
    degrees = [degree for node, degree in G.degree()]
    # 计算统计指标
    mean_degree = np.mean(degrees)
    median_degree = np.median(degrees)
    std_degree = np.std(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    
    edge_consistency={}
    fraud_node_set=set()
    for edge in tqdm(G.edges(),desc="Check edge consistent..."):
        node1_class=G.nodes[edge[0]]['Class']
        node2_class=G.nodes[edge[1]]['Class']

        if node1_class==node2_class:
            edge_consistency[edge]=1
        else:
            edge_consistency[edge]=0

    value_counts=Counter(edge_consistency.values())
    ratio=value_counts[0]/G.number_of_edges()
    return G.number_of_nodes(),G.number_of_edges(),density,mean_degree,median_degree,std_degree,max_degree,min_degree,ratio

def save_graph(name, G):
    # save graph object to file
    pickle.dump(G, open("graph/"+name+'.pickle', 'wb'))

def load_graph(name, G):
    # load graph object from file
    G = pickle.load(open("graph/"+name+'.pickle', 'rb'))
    return G

def data_construction(type, cos_sim, cos_sim_dict_train):
    data_dic={'threshold':[],'number_of_nodes':[],'number_of_edges':[],'density':[]
              ,'mean_degree':[],'median_degree':[],'std_of_degree':[],'max_degree':[]
              ,'min_degree':[],'ratio_of_heterogeneous':[]}
    graph=[]
    for factor in range(31):
        # threshold
        threshold=cos_sim_dict_train['median']+factor*cos_sim_dict_train['std']
        data_dic['threshold'].append(threshold)
        G=graph_construction(cos_sim,df,threshold)
        graph.append(G)
        # 存graph
        save_graph(type+"/"+str(threshold), G)
        
        non,noe,d,meand,medd,stdd,maxd,mind,ratio=graph_describe(G,3)
        data_dic['number_of_nodes'].append(non)
        data_dic['number_of_edges'].append(noe)
        data_dic['density'].append(d)
        data_dic['mean_degree'].append(meand)
        data_dic['median_degree'].append(medd)
        data_dic['std_of_degree'].append(stdd)
        data_dic['max_degree'].append(maxd)
        data_dic['min_degree'].append(mind)
        data_dic['ratio_of_heterogeneous'].append(ratio)
    
    return graph, data_dic

# 存成data.json
def save_data_dic(type, data_dic):
    for key in data_dic:
        if isinstance(data_dic[key][0], np.int32) or isinstance(data_dic[key][0], np.int64):
            data_dic[key]=[int(num) for num in data_dic[key]]
        elif isinstance(data_dic[key][0], np.float64):
            data_dic[key]=[float(num) for num in data_dic[key]]
    json_data=json.dumps(data_dic,indent=4)
    with open('data_'+type+'.json','w') as json_file:
        json_file.write(json_data)
    with open("data.json",'r') as json_file:
        data_dic=json.load(json_file)

def read_graph_mix(type):
    # 讀所有graph
    folder_path = './graph/'+type+"/"
    graph_dic = {}
    # 遍歷資料夾下的所有檔案
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # print(file_path)
        # 檢查是否是.pkl檔案
        if filename.endswith(".pickle"):
            # 讀取.pkl檔案
            try:
                with open(file_path, "rb") as file:
                    G = pickle.load(file)
                    # 在這裡可以對讀取的資料進行處理
                    print(f"從 '{file_path}' 讀取到資料：{G}")
                    threshold = float(os.path.splitext(filename)[0])
                    graph_dic[threshold] = G
            except Exception as e:
                print(f"讀取 '{file_path}' 時發生錯誤：{e}")
    return graph_dic

def G_to_data(graph_dic):
    all_data = []
    for threshold in tqdm(sorted(graph_dic.keys())):
        G = graph_dic[threshold]
        x, y, edge_index = [], [], []
        for node in G.nodes():
            x.append(list(G.nodes[node].values())[:-1])
            y.append(list(G.nodes[node].values())[-1])
    
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
    
        data = Data(x=x, edge_index=edge_index, y=y)
        data['threshold'] = threshold
        all_data.append(data)
    return all_data

def rescale(x):
    return ((x + 1) / 2)*(1-(2e-06)) + 1e-06


def train_model(args, data, model, optimizer, loss_function):
    stats = {
        "best_loss": 1e9,
        "best_epoch": -1,
    }
    model.train()

    label_ones =  torch.ones(1, data.x.shape[0]).to(args["device"])
    label_zeros = torch.zeros(1, data.x.shape[0]).to(args["device"])

    for epoch in tqdm(range(args['num_epoch'])):
        optimizer.zero_grad()
        data = data.to(args['device'])
        # forward(gat+linear)
        h_ego, h_neighbor = model(data.x, data.edge_index)
        h_ego_neg, h_neighbor_neg  = model.negative_sample(h_ego, h_neighbor)
        # 算 -c
        c_neighbor_pos = model.discriminator(h_ego, h_neighbor)
        c_neighbor_neg = model.discriminator(h_ego, h_neighbor_neg)
        c_ego_neg = model.discriminator(h_ego, h_ego_neg)
        # rescal(x) = (x-(-1)) / 2，使介於0~1(原介於-1~1)
        score_pos = rescale(c_neighbor_pos)
        score_aug = rescale(c_neighbor_neg)
        score_nod = rescale(c_ego_neg)
        
        # BCE loss
        # ego-neighbor postive, ego-neighbor negative, ego-ego negative
        loss_pos = loss_function(score_pos, label_zeros)
        loss_aug = loss_function(score_aug, label_ones)
        loss_nod = loss_function(score_nod, label_ones)
        
        loss_sum = loss_pos + args['alpha'] * loss_aug  + args['gamma'] * loss_nod

        loss_sum.backward()
        # 只用postive判斷好壞
        if loss_pos < stats["best_loss"]:
            stats["best_loss"] = loss_pos.item()
            stats["best_epoch"] = epoch
            torch.save(model.state_dict(), args['state_path'])
        optimizer.step()

        # if epoch % 100 ==0:
        #     eval_model(args, data, model)


    return stats

def eval_model(args, data, model):
    model.eval()
    with torch.no_grad():
        data = data.to(args["device"])
        h_ego, h_neighbor = model(data.x, data.edge_index)
        c_neighbor_pos = model.discriminator(h_ego, h_neighbor)
        
        y_true = (data.y).detach().cpu().tolist()
        y_score = c_neighbor_pos.squeeze().detach().cpu().tolist()
        auc = roc_auc_score(y_true, y_score)
        precision_n_score = pyod.utils.utility.precision_n_scores(y_true, y_score, n=None)
        
    return auc, precision_n_score

def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(args, data_train, data_test):
    set_random_seeds(args['seed'])
    # Create model
    model = myGNN(args['enc_num_heads'], args['enc_input_dim'], args['enc_hidden_dim'],  args['enc_num_layers'], args["linear_output_dim"])
    model.to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    loss_function = torch.nn.BCELoss()
    # train
    stats = train_model(
        args, data, model, optimizer, loss_function
    )
    # eval
    model.load_state_dict(torch.load(args["state_path"]))
    auc, precision_n_score = eval_model(args, data_test, model)
    stats["AUC"] = auc
    stats["Precision@n"] = precision_n_score

    return model, stats
if __name__ == "__main__":
    df=DataLoader('creditcard.csv')
    df_train, df_eval=data_sampling(df)
    # 標準化
    df_train=data_standardization(df_train)
    df_eval=data_standardization(df_eval)
    # 去掉時間欄位
    df_train=df_train.drop('Time', axis=1)
    df_eval=df_eval.drop('Time', axis=1)
    # 計算余弦相似度
    calculate_cosine_similarity(df_train, "train")
    calculate_cosine_similarity(df_eval, "eval")
    cos_sim_train=np.load("cos_sim_train.npy")
    cos_sim_eval=np.load("cos_sim_eval.npy")
    # 相似度統計量
    cos_sim_dict_train=cos_sim_describe(cos_sim_train)
    print(cos_sim_dict_train)
    cos_sim_dict_eval =cos_sim_describe(cos_sim_eval)
    print(cos_sim_dict_eval)
    # 圖data
    graph, data_dic = data_construction("train", cos_sim_train, cos_sim_dict_train)
    # !!!!test也用train的相似度去計算threshold
    graph_test, data_dic_test = data_construction("test", cos_sim_eval, cos_sim_dict_train)
    # 儲存data資料
    save_data_dic("train", data_dic)
    save_data_dic("test", data_dic_test)
    # 讀取data
    graph_dic = read_graph_mix("train")
    graph_dic_test = read_graph_mix("test")
    # networkx graph 轉pytorch Data
    all_data = G_to_data(graph_dic)
    all_data_test = G_to_data(graph_dic_test)
    # 參數
    args = {"lr": 5e-4, 
        "alpha": 0.3, 
        "gamma": 0.4, 
        "state_path": "model.pkl", 
        "device": "cuda:0", 
        "seed": 1, 
        "num_epoch": 500, 
        "weight_decay": 0.0, 
        "enc_num_heads": 3, 
        "enc_input_dim":all_data[0].x.shape[1], 
        "enc_hidden_dim": 32, 
        "linear_output_dim": 64,
        "enc_num_layers":1,
       }
    auc_dict = {}
    precision_n_dict = {}
    for data, data_test in zip(all_data, all_data_test): 
        print("----------threshod = "+str(data.threshold)+"-----------------")
        model, stats = run_experiment(args, data, data_test)
        auc_dict[data.threshold]= stats["AUC"]
    sorted_auc_dict = dict(sorted(auc_dict.items(), key=lambda item: item[1], reverse=True))
    print("==================================")
    print("Threshold vs. AUC：")
    print(sorted_auc_dict)
    print("==================================")
    