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

import networkx as nx

import warnings
warnings.filterwarnings("ignore")

import os



seed=0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True



def prepare(df_drug, feature_list,mechanism,action,drugA,drugB):

    d_label = {}
    d_feature = {}

    d_event=[]
    for i in range(len(mechanism)):
        d_event.append(mechanism[i]+" "+action[i])


    count={}
    for i in d_event:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    event_num=len(count)
    list1 = sorted(count.items(), key=lambda x: x[1],reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]]=i


    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  #vector=[]
    for i in feature_list:

        tempvec=feature_vector(i, df_drug)
        vector = np.hstack((vector,tempvec))
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    new_feature = []
    new_label = []

    for i in range(len(d_event)):
        temp=np.hstack((d_feature[drugA[i]],d_feature[drugB[i]]))
        new_feature.append(temp)
        new_label.append(d_label[d_event[i]])

        
    new_feature = np.array(new_feature) #323539*....
    new_label = np.array(new_label)  #323539

    return new_feature, new_label, drugA,drugB,event_num

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
    
    print(feature_name+" len is:"+ str(len(sim_matrix[0])))
    return sim_matrix



class DDIDataset(Dataset):
    def __init__(self,x,y):
        self.len=len(x)
        self.x_data=torch.from_numpy(x)

        self.y_data=torch.from_numpy(y)
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
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
        self.attn = MultiHeadAttention(input_dim,bert_n_heads)

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
    def __init__(self,vector_size):
        super(AE2,self).__init__()
        
        self.vector_size=vector_size//2
        
        self.l1 = torch.nn.Linear(self.vector_size,(self.vector_size+len_after_AE//2)//2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE//2)//2)
        
        self.att2=EncoderLayer((self.vector_size+len_after_AE//2)//2,bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size+len_after_AE//2)//2,len_after_AE//2)
        
        self.l3 = torch.nn.Linear(len_after_AE//2,(self.vector_size+len_after_AE//2)//2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE//2)//2)
        
        self.l4 = torch.nn.Linear((self.vector_size+len_after_AE//2)//2,self.vector_size)
        
        self.dr = torch.nn.Dropout(drop_out_rating)
        
        self.ac=gelu
        
    def forward(self,X):
        
        X1=X[:,0:self.vector_size]
        X2=X[:,self.vector_size:]
        
        X1=self.dr(self.bn1(self.ac(self.l1(X1))))
        X1=self.att2(X1)
        X1=self.l2(X1)
        X_AE1=self.dr(self.bn3(self.ac(self.l3(X1))))
        X_AE1=self.l4(X_AE1)
        
        X2=self.dr(self.bn1(self.ac(self.l1(X2))))
        X2=self.att2(X2)
        X2=self.l2(X2)
        X_AE2=self.dr(self.bn3(self.ac(self.l3(X2))))
        X_AE2=self.l4(X_AE2)
        
        X=torch.cat((X1,X2), 1)
        X_AE=torch.cat((X_AE1,X_AE2), 1)
        
        return X,X_AE


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

        X = X.view(-1, self.vector_size // 2 , 1)

        output, (hn, cn) = self.lstm(X)

        output = self.ac(output[:, -1, :])

        output = self.fc(output)

        output = output.contiguous().view(-1, len_after_AE)

        return output


class ADDAE(torch.nn.Module):
    def __init__(self,vector_size):
        super(ADDAE,self).__init__()

        self.vector_size=vector_size//2

        self.l1 = torch.nn.Linear(self.vector_size,(self.vector_size+len_after_AE)//2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE)//2)

        self.att1=EncoderLayer((self.vector_size+len_after_AE)//2)
        self.l2 = torch.nn.Linear((self.vector_size+len_after_AE)//2,len_after_AE)
        #self.att2=EncoderLayer(len_after_AE//2)

        self.l3 = torch.nn.Linear(len_after_AE,(self.vector_size+len_after_AE)//2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE)//2)

        self.l4 = torch.nn.Linear((self.vector_size+len_after_AE)//2,self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac=gelu

    def forward(self,X):

        X1=X[:,0:self.vector_size]
        X2=X[:,self.vector_size:]
        X=X1+X2

        X=self.dr(self.bn1(self.ac(self.l1(X))))

        X=self.att1(X)
        X=self.l2(X)

        X_AE=self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE=self.l4(X_AE)
        X_AE=torch.cat((X_AE,X_AE), 1)

        return X,X_AE


class BERT(torch.nn.Module):
    def __init__(self,input_dim,n_heads,n_layers,event_num):
        super(BERT, self).__init__()
        
        self.vae1 = VAE1(input_dim,32)
        self.ae2 = AE2(input_dim)  # twin loss
        self.lstm = LSTMModel(input_dim, hidden_size=64, num_layers=2)
        self.ADDAE = ADDAE(input_dim)
        
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.input_dim=input_dim
        
        self.layers = torch.nn.ModuleList([EncoderLayer(len_after_AE*5,n_heads) for _ in range(n_layers)])
        self.AN=torch.nn.LayerNorm(len_after_AE*5)
        
        self.l1=torch.nn.Linear(len_after_AE*5,(len_after_AE*5+event_num)//2)
        self.bn1=torch.nn.BatchNorm1d((len_after_AE*5+event_num)//2)
        
        self.l2=torch.nn.Linear((len_after_AE*5+event_num)//2,event_num)
        
        self.ac=gelu
        
    def forward(self, X):
        X1, X_VAE1 = self.vae1(X)
        X2, X_AE2 = self.ae2(X)
        X3 = self.lstm(X)
        X4, X_AE4 = self.ADDAE(X)

        X5=X1+X2+X3+X4
        
        X=torch.cat((X1,X2,X3,X4,X5), 1)
        
        for layer in self.layers:
            X = layer(X)
        X=self.AN(X)
        
        X=self.dr(self.bn1(self.ac(self.l1(X))))
        
        X=self.l2(X)
        
        return X,X_VAE1,X_AE2,X_AE4


class focal_loss(nn.Module):
    def __init__(self, gamma=2):
        
        super(focal_loss,self).__init__()
        
        self.gamma = gamma

    def forward(self, preds, labels):
        labels = labels.view(-1, 1) # [B * S, 1]
        preds = preds.view(-1, preds.size(-1)) # [B * S, C]
        
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1, labels)
        preds_logsoft = preds_logsoft.gather(1, labels)
        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        
        loss = loss.mean()
        
        return loss
class my_loss1(nn.Module):
    def __init__(self):
        
        super(my_loss1,self).__init__()
        
        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.criteria2=torch.nn.MSELoss()

    def forward(self, X, target,inputs,X_AE1,X_AE2,X_AE4):


        
        loss=calssific_loss_weight*self.criteria1(X,target)+\
             self.criteria2(inputs.float(),X_AE1)+\
             self.criteria2(inputs.float(),X_AE2)+\
             self.criteria2(inputs.float(),X_AE4)
        return loss
class my_loss2(nn.Module):
    def __init__(self):
        
        super(my_loss2,self).__init__()
        
        self.criteria1 = focal_loss()
        self.criteria2=torch.nn.MSELoss()

    def forward(self, X, target,inputs,X_AE1,X_AE2,X_AE4):

        loss=calssific_loss_weight*self.criteria1(X,target)+\
             self.criteria2(inputs.float(),X_AE1)+\
             self.criteria2(inputs.float(),X_AE2)+\
             self.criteria2(inputs.float(),X_AE4)
        return loss




def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y

def BERT_train(model,x_train,y_train,x_test,y_test,event_num):

    model_optimizer=RAdam(model.parameters(),lr=learn_rating,weight_decay=weight_decay_rate)
    model=torch.nn.DataParallel(model)
    model=model.to(device)

    x_train=np.vstack((x_train,np.hstack((x_train[:,len(x_train[0])//2:],x_train[:,:len(x_train[0])//2]))))
    y_train = np.hstack((y_train, y_train))
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    len_train=len(y_train)
    len_test=len(y_test)
    print("arg train len", len(y_train))
    print("test len", len(y_test))


    train_dataset = DDIDataset(x_train,np.array(y_train))
    test_dataset = DDIDataset(x_test,np.array(y_test))
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


    for epoch in range(epo_num):
        if epoch<epoch_changeloss:
            my_loss=my_loss1()
        else:
            my_loss=my_loss2()
        
        running_loss = 0.0
        
        model.train()
        for batch_idx,data in enumerate(train_loader,0):
            x, y = data

            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(x.size()[0]).cuda()
            inputs=lam * x + (1 - lam) * x[index, :]

            targets_a, targets_b = y, y[index]

            inputs=inputs.to(device)
            targets_a=targets_a.to(device)
            targets_b = targets_b.to(device)
            
            model_optimizer.zero_grad()     
            #forward + backward+update
            X,X_AE1,X_AE2,X_AE4=model(inputs.float())


            loss=lam * my_loss(X, targets_a,inputs,X_AE1,X_AE2,X_AE4)+(1-lam)*my_loss(X, targets_b,inputs,X_AE1,X_AE2,X_AE4)


            loss.backward()
            model_optimizer.step()   
            running_loss += loss.item()


    pre_score=np.zeros((0, event_num), dtype=float)
    model.eval()        
    with torch.no_grad():
        for batch_idx,data in enumerate(test_loader,0):
            inputs,_=data
            inputs=inputs.to(device)
            X, _, _, _ = model(inputs.float())
            pre_score =np.vstack((pre_score,F.softmax(X).cpu().numpy()))
    return pre_score


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, np.arange(event_num))

    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    # result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
    # result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    # result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[3] = f1_score(y_test, pred_type, average='macro')
    # result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[4] = precision_score(y_test, pred_type, average='macro')
    # result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[5] = recall_score(y_test, pred_type, average='macro')

    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        # result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average=None)
        result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]


def cross_val(feature,label,drugA,drugB,event_num):

    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])


    temp_drug1 = [[] for i in range(event_num)]
    temp_drug2 = [[] for i in range(event_num)]
    for i in range(len(label)):
        temp_drug1[label[i]].append(drugA[i])
        temp_drug2[label[i]].append(drugB[i])
    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drug1[i])):
            drug_cro_dict[temp_drug1[i][j]] = j % cross_ver_tim
            drug_cro_dict[temp_drug2[i][j]] = j % cross_ver_tim
    train_drug = [[] for i in range(cross_ver_tim)]
    test_drug = [[] for i in range(cross_ver_tim)]
    for i in range(cross_ver_tim):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)


    
    for cross_ver in range(cross_ver_tim):
        
        model=BERT(len(feature[0]),bert_n_heads,bert_n_layers,event_num)

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        for i in range(len(drugA)):
            if (drugA[i] in np.array(train_drug[cross_ver])) and (drugB[i] in np.array(train_drug[cross_ver])):
                X_train.append(feature[i])
                y_train.append(label[i])

            if (drugA[i] not in np.array(train_drug[cross_ver])) and (drugB[i] not in np.array(train_drug[cross_ver])):
                X_test.append(feature[i])
                y_test.append(label[i])


        print("train len", len(y_train))
        print("test len", len(y_test))

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        
        pred_score=BERT_train(model,X_train,y_train,X_test,y_test,event_num)
        
        pred_type = np.argmax(pred_score, axis=1)
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))

        y_true = np.hstack((y_true, y_test))
        
    result_all, result_eve= evaluate(y_pred, y_score, y_true, event_num)

    return result_all, result_eve


file_path="./"

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_n_heads=4
bert_n_layers=4
drop_out_rating=0.5
batch_size=512
len_after_AE=700
learn_rating=0.000005
epo_num=120
cross_ver_tim=5
cov2KerSize=50
cov1KerSize=25
calssific_loss_weight=5
epoch_changeloss=epo_num//2
weight_decay_rate=0.0001
feature_list = ["smile","target","enzyme"]


def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task3'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


def main():
    
    conn = sqlite3.connect("./event.db")
    
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    

    
    new_feature, new_label, drugA,drugB,event_num=prepare(df_drug,feature_list,mechanism,action,drugA,drugB)
    np.random.seed(seed)
    np.random.shuffle(new_feature)
    np.random.seed(seed)
    np.random.shuffle(new_label)
    np.random.seed(seed)
    np.random.shuffle(drugA)
    np.random.seed(seed)
    np.random.shuffle(drugB)
    print("dataset len", len(new_feature))
    
    start=time.time()
    result_all, result_eve=cross_val(new_feature,new_label,drugA,drugB,event_num,)
    print("time used:", (time.time() - start) / 3600)
    save_result(file_path,"all",result_all)
    save_result(file_path,"each",result_eve)



main()

