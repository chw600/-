from __future__ import print_function
from utils import create_adj_s, create_adj_t
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class CustomDataset(Dataset):
    def __init__(self, data_name, file_path, time_windows, data_dim, step=1, mode='Train'):
        '''
        :param time_windows: 时间窗长度
        :param step: 时间窗步长
        :param mode: 读取训练集还是测试集，train为训练集，别的为测试集
        '''
        # if mode == 'Train':
        #     print('Train mode')
        # else:
        #     print('Test mode')
        print("------Processing "+data_name+"------")
        scaler = MinMaxScaler(feature_range=(0, 1))
        normal_data = np.loadtxt(file_path, delimiter=',')
        scaler.fit(normal_data.reshape(-1, data_dim))
        # 用正常数据的最大值最小值做归一化
        normalized_data = scaler.transform(normal_data.reshape(-1, data_dim))
        torch_data = torch.FloatTensor(normalized_data)
        self.inout_seq = []
        self.data_length = len(torch_data)
        print(torch_data.shape)
        # 左闭右开
        # for i in range(0, (self.data_length - time_windows) // step + 1, step):
        for i in range(0, self.data_length // time_windows + 1):
            data_seq = torch_data[i:i + time_windows]
            if mode == 'Train':  # 用正常数据训练
                self.inout_seq.append(data_seq)
            else:  # 用带异常的数据测试
                self.inout_seq.append(data_seq)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        sample = self.inout_seq[index]
        return sample

    def __len__(self):
        return len(self.inout_seq)

# 基于预测的方法的detaset
class PredictorDataset(Dataset):
    def __init__(self, data_name, file_path, time_windows, data_dim, step=1, mode='Train'):
        '''
        :param csv_file: 数据集
        :param time_windows: 时间窗长度
        :param step: 时间窗步长
        :param mode: 读取训练集还是测试集，train为训练集，别的为测试集
        '''
        # if mode == 'Train':
        #     print('Train mode')
        # else:
        #     print('Test mode')
        print("------Processing "+data_name+"------")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        normal_data = np.loadtxt(file_path, delimiter=',')
        scaler.fit(normal_data.reshape(-1, data_dim))
        # 用正常数据的最大值最小值做归一化
        normalized_data = scaler.transform(normal_data.reshape(-1, data_dim))
        torch_data = torch.FloatTensor(normalized_data)
        self.inout_seq = []
        self.predict_seq = []
        self.data_length = torch_data.size(0)
        # 左闭右开
        for i in range(0, (self.data_length - time_windows) // step, step):
            data_seq = torch_data[i:i + time_windows]
            predict = torch_data[i + 1:i + time_windows + 1]
            if mode == 'Train':  # 用正常数据训练
                self.inout_seq.append(data_seq)
                self.predict_seq.append(predict)
            else:  # 用带异常的数据测试
                self.inout_seq.append(data_seq)
                self.predict_seq.append(predict)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        sample = self.inout_seq[index]
        predict = self.predict_seq[index]
        return sample, predict

    def __len__(self):
        return len(self.predict_seq)

def Dealing_data(file_path):

    scaler = MinMaxScaler(feature_range=(0, 1))
    normal_data = np.loadtxt(file_path, delimiter=',')
    scaler.fit(normal_data.reshape(-1, normal_data.shape[-1]))
    # 用正常数据的最大值最小值做归一化
    normalized_data = scaler.transform(normal_data.reshape(-1, normal_data.shape[-1]))
    torch_data = torch.FloatTensor(normalized_data)

    return torch_data