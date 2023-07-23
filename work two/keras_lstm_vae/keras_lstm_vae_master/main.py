from Choosing_steps import Choosing_steps
import random
import networkx as nx
import numpy as np
import torch
import os
from utils import create_adj_s, create_adj_t
import matplotlib.pyplot as plt


def mkdir(path):
    path = path.strip()
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

if __name__ == "__main__":
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # nx.adjacency_matrix()
    # timewindows = 2
    a = np.array([[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165],
                 [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180],
                 [30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195],
                 [45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210],
                 [60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225],
                 [75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240],
                 [90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255],
                 [105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270],
                 [120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285],
                 [135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300],
                 [150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315],
                 [165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330]])
    b = np.array([[30, 46, 67, 75, 90, 105, 120, 139, 159, 154, 180, 195],
                 [45, 70, 75, 90, 105, 120, 135, 141, 169, 120, 195, 210],
                 [60, 78, 98, 105, 120, 135, 150, 157, 189, 143, 210, 225],
                 [75, 99, 115, 120, 135, 150, 165, 177, 199, 243, 225, 240],
                 [90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255],
                 [105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270],
                 [120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285],
                 [135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300],
                 [150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315],
                 [165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330],
                 [180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345],
                 [195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360]])
    c = np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1])

    x_train = 10 * np.cos(a * np.pi / 180)+10
    x_train = np.transpose(x_train, (1, 0))
    x_train = np.around(x_train, decimals=6)

    y_train = 10 * np.cos(b * np.pi / 180)+10
    y_train = np.transpose(y_train, (1, 0))
    y_train = np.around(y_train, decimals=6)

    label = np.expand_dims(c, 0)
    label = np.around(label, decimals=6)
    train_path = 'cos/train.txt'
    test_path = 'cos/test.txt'
    label_path = 'cos/label.txt'
    np.savetxt(train_path, x_train, fmt='%1.6f', delimiter=',')
    np.savetxt(test_path, y_train, fmt='%1.6f', delimiter=',')
    np.savetxt(label_path, label, fmt='%1.6f', delimiter=',')

    #
    # y_train_2 = 10 * np.cos(c * np.pi / 180)
    # y_train_2 = np.expand_dims(y_train_2, 0)
    # y_train_2 = y_train_2.reshape(1, -1)
    # y_train_2 = np.repeat(y_train_2, 4, axis=0)
    # y_train_2 = y_train_2.reshape(-1, 1)
    # y_train_2 = np.expand_dims(y_train_2, 0)
    # y_train_2 = np.repeat(y_train_2, 1, axis=0)
    #
    # y_train = np.concatenate((y_train_1, y_train_2), axis=-1)
    #
    # print(x_train.shape, y_train.shape)
    # z_train = x_train[:, 0:2, :]
    # for i in range(1, x_train.shape[1] - 1):
    #     z_train = np.concatenate((z_train, x_train[:, i:i + timewindows, :]), axis=0)
    # y_train = np.transpose(y_train, (1, 2, 0))
    # y_train = y_train[:-1, :, :]
    # z_train = np.repeat(z_train, 4, axis=2)
    # y_train = np.repeat(y_train, 4, axis=2)
    # z_test = y_train
    # print(z_train.shape, y_train.shape)

    # train_path = 'ServerMachineDataset/train/machine-1-3.txt'
    # test_path = 'ServerMachineDataset/test/machine-1-3.txt'
    # train_path = '../save/BIGRU/SMD/H_train.txt'
    # train_data = np.loadtxt(train_path)
    # print("-----Processing train_adj_t-----")
    # train_adj_s = create_adj_s(train_data, 0.8)
    # train_path_s = '../save/BIGRU/SMD/H_train_t.txt'
    # np.savetxt(train_path_s, train_adj_s)
    #训练BIGRU模型
    nums_1 = Choosing_steps(file_train=train_path,
                            file_valid=train_path,
                            file_test=test_path,
                            file_label=label_path,
                            gru_dim=12,
                            data_name="cos",
                            time_dim=12,
                            gru_input_dim=1,
                            timewindows=4,
                            sensor_dim=12,
                            steps=1,
                            BiGru_epochs=10,
                            MSST_GCN_epochs=10)
    nums_1()
    #加载BIGRU模型处理train_data, valid_data和test_data，生成隐变量H
    nums_2 = Choosing_steps(file_train=train_path,
                            file_valid=train_path,
                            file_test=test_path,
                            file_label=label_path,
                            gru_dim=12,
                            data_name="cos",
                            time_dim=12,
                            gru_input_dim=1,
                            timewindows=4,
                            sensor_dim=12,
                            steps=2,
                            BiGru_epochs=10,
                            MSST_GCN_epochs=10)
    nums_2()
    #加载train_data，valid_data和test_data的隐变量H，生成相应的邻接矩阵
    nums_3 = Choosing_steps(file_train=train_path,
                            file_valid=train_path,
                            file_test=test_path,
                            file_label=label_path,
                            gru_dim=12,
                            data_name="cos",
                            time_dim=12,
                            gru_input_dim=1,
                            timewindows=4,
                            sensor_dim=12,
                            steps=3,
                            BiGru_epochs=10,
                            MSST_GCN_epochs=10)
    nums_3()
    #训练MSST_GCN模型
    nums_4 = Choosing_steps(file_train=train_path,
                            file_valid=train_path,
                            file_test=test_path,
                            file_label=label_path,
                            gru_dim=12,
                            data_name="cos",
                            time_dim=12,
                            gru_input_dim=1,
                            timewindows=4,
                            sensor_dim=12,
                            steps=4,
                            BiGru_epochs=10,
                            MSST_GCN_epochs=10)
    nums_4()
    #加载MSST_GCN模型处理测试数据
    nums_5 = Choosing_steps(file_train=train_path,
                            file_valid=train_path,
                            file_test=test_path,
                            file_label=label_path,
                            gru_dim=12,
                            data_name="cos",
                            time_dim=12,
                            gru_input_dim=1,
                            timewindows=4,
                            sensor_dim=12,
                            steps=5,
                            BiGru_epochs=10,
                            MSST_GCN_epochs=10)
    nums_5()
    # plt.plot(z_train, label='data')
    # plt.plot(preds, label='predict')
    # plt.legend()
    # plt.show()
# x = torch.randn(32, 10, 64)
# z, H = BIGRU(64, 32)(x)
# print(z.shape)
# print(H.shape)

