# coding: utf-8

from numpy.random import seed
import csv
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import scipy.sparse as sp
import math
import copy

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA

import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from pytorchtools import BalancedDataParallel
from radam import RAdam
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")

import os
print(torch.cuda.is_available())

file_path="./"

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_n_heads=4
bert_n_layers=4
drop_out_rating=0.3
batch_size=256
len_after_AE=500
learn_rating=0.00001
epo_num=120
cross_ver_tim=5
cov2KerSize=50
cov1KerSize=25
calssific_loss_weight=5
epoch_changeloss=epo_num//2
weight_decay_rate=0.0001
feature_list = ["smile","target","enzyme"]

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def prepare(df_drug, feature_list, mechanism, action, drugA, drugB):
    # d_label：用于存储药物相互作用事件的标签（转换为数字）。
    # d_feature：用于存储药物的特征向量。
    # d_event：用于存储药物相互作用事件的字符串形式。
    d_label = {}
    d_feature = {}
    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    count = {}   # 统计每个药物相互作用事件的概率
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    event_num = len(count)    #event_num 药物相互作用事件的数量
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  # vector=[]
    for i in feature_list:

        tempvec = feature_vector(i, df_drug)
        vector = np.hstack((vector, tempvec))
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    new_feature = []
    new_label = []

    for i in range(len(d_event)):
        temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
        new_feature.append(temp)
        new_label.append(d_label[d_event[i]])

    new_feature = np.array(new_feature)
    new_label = np.array(new_label)  #

    return new_feature, new_label, event_num, d_label


def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()

    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0])))
    return sim_matrix


class DDIDataset(Dataset):
    def __init__(self, x, y):
        self.len = len(x)
        self.x_data = torch.from_numpy(x)

        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class DDIDataset_test(Dataset):
    def __init__(self,x):
        self.len=len(x)
        self.x_data=torch.from_numpy(x)
        
    def __getitem__(self,index):
        return self.x_data[index]
    def __len__(self):
        return self.len


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, n_heads, output_dim=None, dropout_rate=0.1):

        super(MultiHeadAttention, self).__init__()

        self.d_k = self.d_v = input_dim // n_heads

        self.n_heads = n_heads

        if output_dim is None:

            self.output_dim = input_dim
        else:

            self.output_dim = output_dim

        self.W_Q = nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)

        self.W_K = nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)

        self.W_V = nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)

        self.fc = nn.Linear(self.n_heads * self.d_v, self.output_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X, attn_mask=None):

        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)

        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)

        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, float('-1e20'))

        attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)

        output = self.fc(context)

        return output

class AdaptiveResidualLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveResidualLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, input_dim)
        self.alpha = torch.nn.Parameter(torch.tensor(0.6))  # learnable weight

    def forward(self, X):
        residual = X
        output = self.linear(X)
        return output + self.alpha * residual


class SwiGLU(torch.nn.Module):
    def __init__(self, input_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.linear2 = torch.nn.Linear(input_dim, input_dim)

    def forward(self, X):
        return self.linear1(X) * torch.sigmoid(self.linear2(X))


class AttentionLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, bert_n_heads)

    def forward(self, X):
        return X + self.attn(X)  # 直接加上多头注意力输出


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(EncoderLayer, self).__init__()
        self.adaptive_residual = AdaptiveResidualLayer(input_dim)
        self.swi_glu = SwiGLU(input_dim)
        self.attention = AttentionLayer(input_dim)

    def forward(self, X):
        X = self.adaptive_residual(X)
        X = self.swi_glu(X)
        X = self.attention(X)
        return X


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class VAE1(nn.Module):

    def __init__(self, vector_size, latent_dim):
        super(VAE1, self).__init__()

        self.vector_size = vector_size
        self.latent_dim = latent_dim
        self.l1 = nn.Linear(vector_size, (vector_size + len_after_AE) // 2)
        self.bn1 = nn.BatchNorm1d((vector_size + len_after_AE) // 2)
        self.att2 = EncoderLayer((self.vector_size + len_after_AE) // 2)
        self.fc_mu = nn.Linear((vector_size + len_after_AE) // 2, latent_dim)
        self.fc_logvar = nn.Linear((vector_size + len_after_AE) // 2, latent_dim)

        self.l3 = nn.Linear(latent_dim, (vector_size + len_after_AE) // 2)
        self.bn3 = nn.BatchNorm1d((vector_size + len_after_AE) // 2)
        self.l4 = nn.Linear((vector_size + len_after_AE) // 2, vector_size)
        self.dr = nn.Dropout(0.2)

    def encode(self, x):
        x = self.dr(self.bn1(F.gelu(self.l1(x))))
        x = self.att2(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.dr(self.bn3(F.gelu(self.l3(z))))
        z = self.l4(z)

        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)

        return x, x_reconstructed


class AE2(torch.nn.Module):
    def __init__(self, vector_size):
        super(AE2, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE // 2) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE // 2) // 2)

        self.att2 = EncoderLayer((self.vector_size + len_after_AE // 2) // 2)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE // 2) // 2, len_after_AE // 2)

        self.l3 = torch.nn.Linear(len_after_AE // 2, (self.vector_size + len_after_AE // 2) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE // 2) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE // 2) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]

        X1 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X1 = self.att2(X1)
        X1 = self.l2(X1)
        X_AE1 = self.dr(self.bn3(self.ac(self.l3(X1))))
        X_AE1 = self.l4(X_AE1)

        X2 = self.dr(self.bn1(self.ac(self.l1(X2))))
        X2 = self.att2(X2)
        X2 = self.l2(X2)
        X_AE2 = self.dr(self.bn3(self.ac(self.l3(X2))))
        X_AE2 = self.l4(X_AE2)

        X = torch.cat((X1, X2), 1)
        X_AE = torch.cat((X_AE1, X_AE2), 1)

        return X, X_AE


class LSTMModel(nn.Module):

    def __init__(self, vector_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()

        self.vector_size = vector_size

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, len_after_AE)

        self.ac = nn.GELU()

    def forward(self, X):
        X1 = X[:, 0:self.vector_size // 2]

        X2 = X[:, self.vector_size // 2:]

        X = torch.cat((X1, X2), 0)

        X = X.view(-1, self.vector_size // 2, 1)

        output, (hn, cn) = self.lstm(X)

        output = self.ac(output[:, -1, :])

        output = self.fc(output)
        output = output.contiguous().view(-1, len_after_AE)

        return output

class ADDAE(torch.nn.Module):
    def __init__(self, vector_size):
        super(ADDAE, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.att1 = EncoderLayer((self.vector_size + len_after_AE) // 2)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)

        self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]
        X = X1 + X2

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.att1(X)
        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)
        X_AE = torch.cat((X_AE, X_AE), 1)

        return X, X_AE


class BERT(torch.nn.Module):
    def __init__(self, input_dim, n_heads, n_layers, event_num):
        super(BERT, self).__init__()

        self.vae1 = VAE1(input_dim, 32)
        self.ae2 = AE2(input_dim)
        self.lstm = LSTMModel(input_dim, hidden_size=64, num_layers=2)
        self.ADDAE = ADDAE(input_dim)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.input_dim = input_dim

        self.layers = torch.nn.ModuleList([EncoderLayer(len_after_AE * 5) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(len_after_AE * 5)

        self.l1 = torch.nn.Linear(len_after_AE * 5, (len_after_AE * 5 + event_num) // 2)
        self.bn1 = torch.nn.BatchNorm1d((len_after_AE * 5 + event_num) // 2)

        self.l2 = torch.nn.Linear((len_after_AE * 5 + event_num) // 2, event_num)

        self.ac = gelu

    def forward(self, X):
        X1, X_VAE1 = self.vae1(X)
        X2, X_AE2 = self.ae2(X)
        X3 = self.lstm(X)
        X4, X_AE4 = self.ADDAE(X)

        X5 = X1 + X2 + X3 + X4

        X = torch.cat((X1, X2, X3, X4, X5), 1)

        for layer in self.layers:
            X = layer(X)
        X = self.AN(X)

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.l2(X)

        return X, X_VAE1, X_AE2, X_AE4


class focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(focal_loss, self).__init__()

        self.gamma = gamma

    def forward(self, preds, labels):
        labels = labels.view(-1, 1)  # [B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]

        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels)
        preds_logsoft = preds_logsoft.gather(1, labels)

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss


class my_loss1(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()

        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4)
        return loss


class my_loss2(nn.Module):
    def __init__(self):
        super(my_loss2, self).__init__()

        self.criteria1 = focal_loss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4)
        return loss


def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y

def BERT_train(model, x_train, y_train):
    model_optimizer = RAdam(model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate)
    model = torch.nn.DataParallel(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x_train = np.vstack((x_train, np.hstack((x_train[:, len(x_train[0]) // 2:], x_train[:, :len(x_train[0]) // 2]))))
    y_train = np.hstack((y_train, y_train))
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    len_train = len(y_train)

    print("arg train len", len(y_train))


    train_dataset = DDIDataset(x_train, np.array(y_train))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epo_num):
        if epoch < epoch_changeloss:
            my_loss = my_loss1()
        else:
            my_loss = my_loss2()

        running_loss = 0.0

        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
            x, y = data
            #print(x.shape, y.shape)
            
            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(x.size()[0]).cuda()
            #index = torch.randperm(x.size()[0]).to(device)
            inputs = lam * x + (1 - lam) * x[index, :]

            targets_a, targets_b = y, y[index]

            inputs = inputs.to(device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)

            model_optimizer.zero_grad()

            X, X_AE1, X_AE2, X_AE4 = model(inputs.float())

            loss = lam * my_loss(X, targets_a, inputs, X_AE1, X_AE2, X_AE4) + (1 - lam) * my_loss(X, targets_b, inputs,
                                                                                                  X_AE1, X_AE2, X_AE4)

            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()
            print(f"Processed batch {batch_idx}/{len(train_loader)}")  

        print(f'Epoch [{epoch+1}/{epo_num}] completed.')  
        print('epoch [%d] loss: %.6f' % (epoch+1,running_loss/len_train/2))
    torch.save(model.state_dict(), file_path+"case_study_model.pt")


def main():
    
    conn = sqlite3.connect("event/event.db")
    df_drug = pd.read_sql('select * from drug;', conn)
    #print(df_drug)
    df_drug.to_csv('df_drug.csv', index=False, encoding='utf-8')

    extraction = pd.read_sql('select * from extraction;', conn)
    #print(extraction)
    extraction.to_csv('extraction.csv', index=False, encoding='utf-8')

    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    
    drugA_test=[]
    drugB_test=[]
    drug_drug_fea=[]
    drug_drug_score=[]
    drug_drug_type=[]
    train_drug_com=[]
    for i in range(len(drugA)):
        train_drug_com.append(drugA[i]+drugB[i])
        train_drug_com.append(drugB[i]+drugA[i])
    #print(train_drug_com)


    new_feature, new_label, event_num, d_label=prepare(df_drug,feature_list,mechanism,action,drugA,drugB)

    print("药物对的特征向量:",new_feature)
    print("每个药物对的标签:", new_label)
    print("不同药物相互作用事件的总数:", event_num)

    model=BERT(len(new_feature[0]),bert_n_heads,bert_n_layers,event_num)
    #BERT_train(model,new_feature,new_label)
    
    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  #vector=[]用于存储每种药物的特征向量。
    d_feature={}

    for i in feature_list:
        vector = np.hstack((vector,feature_vector(i, df_drug)))

    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    for i in range(len(np.array(df_drug['name']).tolist())):

        for j in range(i+1,len(np.array(df_drug['name']).tolist())):

            if (df_drug['name'][i]+df_drug['name'][j]) not in train_drug_com:

                drugA_test.append(df_drug['name'][i])
                drugB_test.append(df_drug['name'][j])
                drug_drug_fea.append(np.hstack((d_feature[df_drug['name'][i]],d_feature[df_drug['name'][j]])))
    print("测试集中药物对的数量:",len(drug_drug_fea))
            
    
    the_model = BERT(len(new_feature[0]),bert_n_heads,bert_n_layers,event_num)
    the_model = torch.nn.DataParallel(the_model) 
    the_model.load_state_dict(torch.load(file_path+"case_study_model.pt"))
    d_label_inverse = {v: k for k, v in d_label.items()}
    print("d_label_inverse:",d_label_inverse) 

    # 测试数据
    drug_drug_fea = np.array(drug_drug_fea)
    test_dataset = DDIDataset_test(drug_drug_fea)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 模型预测
    for batch_idx, data in enumerate(test_loader, 0):
        inputs = data
        inputs = inputs.to(device)
        X, _, _, _ = the_model(inputs.float())
        X = F.softmax(X).detach().cpu().numpy()
        drug_drug_type.append(np.argmax(X, axis=1).tolist())
        drug_drug_score.append(np.amax(X, axis=1).tolist())
        print(f"Processed batch {batch_idx}/{len(test_loader)}") 

    drug_drug_score=[i for j in drug_drug_score for i in j]
    drug_drug_type=[i for j in drug_drug_type for i in j]
    

    datalist = []
    datalist.append(drugA_test)
    datalist.append(drugB_test)
    datalist.append(drug_drug_score)
    datalist.append(drug_drug_type)

    dataarray = np.array(datalist)
    dataarray = dataarray.T
    #print("dataarry:", dataarray)

    # 重点关注了频率最高的10个事件（共65个），并检查了与每个事件相关的前30个预测。
    num_types = 10

    for i in range(num_types):
        globals()[f'type{i}_drugA'] = []
        globals()[f'type{i}_drugB'] = []
        globals()[f'type{i}_score'] = []
        globals()[f'type{i}_event'] = []

    for i in range(len(dataarray)):
        if dataarray[i][3] == '0':
            type0_drugA.append(dataarray[i][0])
            type0_drugB.append(dataarray[i][1])
            type0_score.append(float(dataarray[i][2]))
            type0_event.append(d_label_inverse[int(dataarray[i][3])])
        elif dataarray[i][3] == '1':
            type1_drugA.append(dataarray[i][0])
            type1_drugB.append(dataarray[i][1])
            type1_score.append(float(dataarray[i][2]))
            type1_event.append(d_label_inverse[int(dataarray[i][3])])
        elif dataarray[i][3] == '2':
            type2_drugA.append(dataarray[i][0])
            type2_drugB.append(dataarray[i][1])
            type2_score.append(float(dataarray[i][2]))
            type2_event.append(d_label_inverse[int(dataarray[i][3])])
        elif dataarray[i][3] == '3':
            type3_drugA.append(dataarray[i][0])
            type3_drugB.append(dataarray[i][1])
            type3_score.append(float(dataarray[i][2]))
            type3_event.append(d_label_inverse[int(dataarray[i][3])])
        elif dataarray[i][3] == '4':
            type4_drugA.append(dataarray[i][0])
            type4_drugB.append(dataarray[i][1])
            type4_score.append(float(dataarray[i][2]))
            type4_event.append(d_label_inverse[int(dataarray[i][3])])
        elif dataarray[i][3] == '5':
            type5_drugA.append(dataarray[i][0])
            type5_drugB.append(dataarray[i][1])
            type5_score.append(float(dataarray[i][2]))
            type5_event.append(d_label_inverse[int(dataarray[i][3])])
        elif dataarray[i][3] == '6':
            type6_drugA.append(dataarray[i][0])
            type6_drugB.append(dataarray[i][1])
            type6_score.append(float(dataarray[i][2]))
            type6_event.append(d_label_inverse[int(dataarray[i][3])])
        elif dataarray[i][3] == '7':
            type7_drugA.append(dataarray[i][0])
            type7_drugB.append(dataarray[i][1])
            type7_score.append(float(dataarray[i][2]))
            type7_event.append(d_label_inverse[int(dataarray[i][3])])
        elif dataarray[i][3] == '8':
            type8_drugA.append(dataarray[i][0])
            type8_drugB.append(dataarray[i][1])
            type8_score.append(float(dataarray[i][2]))
            type8_event.append(d_label_inverse[int(dataarray[i][3])])
        elif dataarray[i][3] == '9':
            type9_drugA.append(dataarray[i][0])
            type9_drugB.append(dataarray[i][1])
            type9_score.append(float(dataarray[i][2]))
            type9_event.append(d_label_inverse[int(dataarray[i][3])])
               

    def save_to_csv(drugA, drugB, score, event, filename):
        datalist = []
        datalist.append(drugA)
        datalist.append(drugB)
        datalist.append(score)
        datalist.append(event)
        dataarray = np.array(datalist)
        dataarray = dataarray.T
        dataarray = dataarray[np.lexsort(dataarray.T)]
        dataarray = dataarray[-30:, :]
        with open(file_path + filename, "w", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["drugA", "drugB", "score", "event"])
            writer.writerows(dataarray)

    save_to_csv(type0_drugA, type0_drugB, type0_score, type0_event, "case_study_type0.csv")
    save_to_csv(type1_drugA, type1_drugB, type1_score, type1_event, "case_study_type1.csv")
    save_to_csv(type2_drugA, type2_drugB, type2_score, type2_event, "case_study_type2.csv")
    save_to_csv(type3_drugA, type3_drugB, type3_score, type3_event, "case_study_type3.csv")
    save_to_csv(type4_drugA, type4_drugB, type4_score, type4_event, "case_study_type4.csv")
    save_to_csv(type5_drugA, type5_drugB, type5_score, type5_event, "case_study_type5.csv")
    save_to_csv(type6_drugA, type6_drugB, type6_score, type6_event, "case_study_type6.csv")
    save_to_csv(type7_drugA, type7_drugB, type7_score, type7_event, "case_study_type7.csv")
    save_to_csv(type8_drugA, type8_drugB, type8_score, type8_event, "case_study_type8.csv")
    save_to_csv(type9_drugA, type9_drugB, type9_score, type9_event, "case_study_type9.csv")
      
      

main()
