import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# from Wave import createSensorWave
# import random


def masked_loss(out, label, mask):
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc


def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte().bool()
    i = x._indices()  # [2, 49216]
    v = x._values()  # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1. / (1 - rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res


# 划分数据集，shanghai.pkl和shanghai_extra.pkl是原数据集
def split_dataset(filename, split_rate=0.5):
    f = open(filename, 'rb')
    data = pickle.load(f)
    index = int(data.shape[0] * split_rate)
    train_data = data[:index]
    test_data = data[index:]
    fs_train = open(filename.split('.')[0] + '_train' + '.pkl', 'wb')
    fs_test = open(filename.split('.')[0] + '_test' + '.pkl', 'wb')
    pickle.dump(train_data, fs_train)
    pickle.dump(test_data, fs_test)
    return


# 添加异常点的函数，从高斯函数中取样，添加至正常点上使其变成异常点
def add_abnormal(filename, rate=0.05, mu=1, sigma=0.1):
    fo = open(filename, 'rb')
    data = pickle.load(fo)
    # [1900, 10, 3]
    abnormal_num = int(data.shape[0] * rate)
    np.random.seed(4)
    # 随机某个时刻异常
    abnormal_index = np.random.choice(data.shape[0], abnormal_num, replace=False)
    # abnormal_index = random.sample(range(0, data.shape[0]), abnormal_num)
    np.random.seed(4)
    sensor_index = np.random.randint(data.shape[1], size=abnormal_num)
    # matrix_index = np.zeros((data.shape[0], 2), int)
    label = np.zeros(data.shape[0], int)
    label[abnormal_index] = 1
    data[abnormal_index, sensor_index] += data[abnormal_index, sensor_index] * 0.1
    print(data.shape, label.shape)
    fs_data = open(filename.split('.')[0] + '_abn' + '.pkl', 'wb')
    fs_label = open(filename.split('.')[0] + '_label' + '.pkl', 'wb')
    pickle.dump(data, fs_data, -1)
    pickle.dump(label, fs_label, -1)
    return data, label



# class SinDataSet(Dataset):
#     def __init__(self, data_name, time_windows, step, mode="Train"):
#         '''
#         :param csv_file: 数据集
#         :param time_windows: 时间窗长度
#         :param step: 时间窗步长
#         :param mode: 读取训练集还是测试集，0为训练集，1为测试集
#         '''
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         if data_name == 'shanghai':
#             if mode == "Train":
#                 print('训练模式')
#                 arr = createSensorWave(timestamp=950, numOfSens=10, numOfFeas=3)
#
#             else:
#                 print('测试模式')
#
#         else:
#
#             print('数据集读取失败')
#
#         self.inout_seq = []
#         self.predict_seq = []
#         self.data_length = torch_data_internal.size(0)
#         print("original data size:",torch_data_internal.size())
#         # 对数据进行归一化，然后再将数据的形状转换回去
#
#         # 左闭右开
#         for i in range(0, self.data_length - time_windows, step):
#
#             # predict = torch_data_internal[i+time_windows]
#             self.inout_seq.append()
#             self.predict_seq.append()
#
#     def __getitem__(self, index):
#         # TODO
#         # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#         # 2. Preprocess the data (e.g. torchvision.Transform).
#         # 3. Return a data pair (e.g. image and label).
#         # 这里需要注意的是，第一步：read one data，是一个data
#         sample = self.inout_seq[index]
#         predict = self.predict_seq[index]
#         return sample, predict
#
#     def __len__(self):
#         return len(self.predict_seq)


# 基于预测的方法的dataset
class PredictorDataset(Dataset):
    def __init__(self, data_name, time_windows, step, mode="Train"):
        '''
        :param csv_file: 数据集
        :param time_windows: 时间窗长度
        :param step: 时间窗步长
        :param mode: 读取训练集还是测试集，0为训练集，1为测试集
        '''
        scaler = MinMaxScaler(feature_range=(0, 1))
        if data_name == 'shanghai':
            if mode == "Train":
                print('训练模式')
                data_internal = pickle.load(open('data/shanghai_train.pkl', 'rb'), encoding='utf-8')
                data_external = pickle.load(open('data/shanghai_extra_train.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                normal_data_external = data_external.copy()
                # 读取邻接矩阵
                adj_path = 'data/shanghai_conj.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            else:
                print('测试模式')
                data_internal = pickle.load(open('data/shanghai_test_abn.pkl', 'rb'), encoding='utf-8')
                data_external = pickle.load(open('data/shanghai_extra_test.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                normal_data_external = data_external.copy()
                # 读取邻接矩阵
                adj_path = 'data/shanghai_conj.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
        else:
            # data_dim = 0
            # values = []
            normal_data_internal = []
            normal_data_external = []
            adj_matrix = []
            print('数据集读取失败')
        # 对数据进行归一化，然后再将数据的形状转换回去
        normalized_data_internal = scaler.fit_transform(
            normal_data_internal.reshape(-1, normal_data_internal.shape[-1]))
        normalized_data_internal = normalized_data_internal.reshape(normal_data_internal.shape)
        normalized_data_external = scaler.fit_transform(
            normal_data_external.reshape(-1, normal_data_external.shape[-1]))
        normalized_data_external = normalized_data_external.reshape(normal_data_external.shape)
        # 转成Tensor
        torch_data_internal = torch.FloatTensor(normalized_data_internal)
        torch_data_external = torch.FloatTensor(normalized_data_external)
        self.inout_seq = []
        self.predict_seq = []
        self.data_length = torch_data_internal.size(0)
        print("original data size:",torch_data_internal.size())
        # 左闭右开
        for i in range(0, self.data_length - time_windows, step):
            data_seq = (torch_data_internal[i:i + time_windows], adj_matrix, torch_data_external[i:i + time_windows])
            # predict = (torch_data_internal[i:i + time_windows + 1], adj_matrix,
            #            torch_data_external[i:i + time_windows + 1])
            predict = torch_data_internal[i+1:i + time_windows + 1]
            # predict = torch_data_internal[i+time_windows]
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def DTWDistance(s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

def create_adj_s(inputs, a):

    T, K = inputs.shape
    A_s = np.zeros(shape=(K, K))
    D_s = np.zeros(shape=(K, K))
    inputs = np.transpose(inputs, (1, 0))
    for k_1 in range(K):
        for k_2 in range(K):
            Similar = np.exp(-DTWDistance(inputs[k_1], inputs[k_2]))
            if(Similar>a):
                A_s[k_1][k_2] = 1
            else:
                A_s[k_1][k_2] = 0
        D_s[k_1][k_1] = np.power(np.sum(A_s[k_1]), -0.5)

    A_s = np.dot(D_s, A_s)
    A_s = np.dot(A_s, D_s)
    return A_s

def create_adj_t(inputs, b_1):

    T, K = inputs.shape
    A_t = np.zeros(shape=(T, T))
    D_t = np.zeros(shape=(T, T))
    for t_1 in range(T):
        for t_2 in range(T):
            Similar = np.sum(inputs[t_1] * inputs[t_2]) \
                          / np.sqrt(np.sum(inputs[t_1] * inputs[t_1])) \
                          / np.sqrt(np.sum(inputs[t_2] * inputs[t_2]))
            if (Similar > b_1):
                A_t[t_1][t_2] = 1
            else:
                A_t[t_1][t_2] = 0
        D_t[t_1][t_1] = np.power(np.sum(A_t[t_1]), -0.5)

    A_t = np.dot(D_t, A_t)
    A_t = np.dot(A_t, D_t)
    return A_t

# split_dataset('data/shanghai.pkl')
# split_dataset('data/shanghai_extra.pkl')
# add_abnormal('data/shanghai_test.pkl')
