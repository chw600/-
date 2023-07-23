from __future__ import print_function
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# 读取雅虎数据集中的数据
def read_yahoo(data_path):
    '''
    :param data_path: 待读取的数据集的路径
    :return: 从数据集中读取出来的数据其对应的标签、和异常点的下标，以numpy形式返回
    '''
    if not os.path.exists(data_path):
        print('file do not exist！')
        return 0
    dataframe = pd.read_csv(data_path)
    value = dataframe['value'].values
    label = dataframe['is_anomaly'].values
    return value, label


def read_kpi(filename):
    # 取前一万个点，每隔一个点下采样一次，参考VAE—GAN论文中的方式
    test_df = pd.HDFStore(filename).get('data')
    test_vals = test_df.value.values[:10000:2].reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    test_vals = scaler.fit_transform(test_vals.reshape(-1, 1))
    # 只取测试集，原因在于LOF、COF、CBLOF、KNN都是无监督方法
    test_labels = test_df.label.values[:10000:2]
    return test_vals, test_labels


# 数据预处理：将数据归一化至-1~1，然后根据滑动窗口和时间步截取数据
def data_preprocessing(data, time_windows, time_step):
    # 将数据归一化至-1~1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    torch_data = torch.FloatTensor(data_normalized).view(-1)
    inout_seq = []
    # 将数据依据滑动时间窗划分,label表示该时间窗是否是异常
    L = len(torch_data)
    for i in range(0, (L - time_windows) // time_step + 1, time_step):
        data_seq = torch_data[i:i + time_windows]
        inout_seq.append(data_seq)
    return inout_seq


# custom weights initialization called on netG and netD
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


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据,shape为(sample_size,feature_size)
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # total0 = torch.cat([source, target], dim=0).cuda()
    # total1 = torch.cat([target, source], dim=0).cuda()
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source = source.view(source.size()[0], -1)
    target = target.view(target.size()[0], -1)
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss


def search_best_f1(encoder_path, generator_path, discriminator_path, test_data, true_label, time_windows):
    '''
    :param encoder_path: encoder参数
    :param generator_path: generator参数
    :param discriminator_path: discriminator参数
    :param test_data: 测试集，shape为:(1,time_windows,data_dim)
    :param true_label: 标签
    :param time_windows: 时间窗长度
    :return:
    '''
    f1_best = 0.
    threshold_best = 0.
    pre_best = 0.
    acc_best = 0.
    recall_best = 0.
    alpha_best = 0.
    encoder = torch.load(encoder_path)
    generator = torch.load(generator_path)
    discriminator = torch.load(discriminator_path)
    encoder.eval()
    generator.eval()
    discriminator.eval()
    for alpha in np.arange(0., 1, 0.05):
        for threshold in np.arange(0.01, 1, 0.01):
            label_detect = []
            for index, data in enumerate(test_data):
                with torch.no_grad():
                    # 取最后一个时间戳的score score = alpha * reconstruct error + (1 - alpha) * discriminator score
                    score = abs(generator(encoder(data)) - data)
                    score = score.view(-1)[-1]
                if score < threshold:
                    label_detect.append(1)
                else:
                    label_detect.append(0)
            f1 = f1_score(true_label[time_windows - 1:], label_detect)
            print(f1)
            if f1 > f1_best:
                f1_best = f1
                pre_best = precision_score(true_label[time_windows - 1:], label_detect)
                acc_best = accuracy_score(true_label[time_windows - 1:], label_detect)
                recall_best = recall_score(true_label[time_windows - 1:], label_detect)
                threshold_best = threshold
                alpha_best = alpha
    print('accuracy:{},precision:{},recall:{},F_1:{}'.format(acc_best, pre_best, recall_best, f1_best, alpha_best))
    return threshold_best


def plot_data(x, data, epoch):
    plt.plot(x, data)
    plt.xlabel("timestamp")
    plt.ylabel("values")
    plt.title("epoch:{}".format(epoch))
    # plt.savefig('./epoch' + str(epoch) + '.png')
    plt.show()
    plt.clf()


def plot_data_withanomaly(file_path):
    time_series_df = pd.read_csv(file_path)
    dataset_value = pd.DataFrame()
    dataset_value['value'] = time_series_df.value.values[700:]
    dataset_value.plot()
    x = np.where(time_series_df.is_anomaly.values[700:] == 1)
    y = [dataset_value['value'][index] for index in x]
    plt.scatter(x, y, s=25, c='r')
    plt.title('real_1_5%.csv')
    plt.show()


# 基于重构的方法的detaset
class CustomDataset(Dataset):
    def __init__(self, data_name, file_path, time_windows, step, mode='Train'):
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
        scaler = MinMaxScaler(feature_range=(0, 1))
        normal_data = np.loadtxt(file_path, delimiter=',')
        data_dim = 38
        scaler.fit(normal_data.reshape(-1, data_dim))
        # 用正常数据的最大值最小值做归一化
        normalized_data = scaler.transform(normal_data.reshape(-1, data_dim))
        torch_data = torch.FloatTensor(normalized_data)
        self.inout_seq = []
        self.data_length = len(torch_data)
        # 左闭右开
        for i in range(0, (self.data_length - time_windows) // step + 1, step):
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
    def __init__(self, data_name, file_path, time_windows, step, mode='Train'):
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
        scaler = MinMaxScaler(feature_range=(-1, 1))
        normal_data = np.loadtxt(file_path, delimiter=',')
        data_dim = 38
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
