from MSST_GCN import MSST_GCNTrainer
from BIGRU import BIGRUTrainer
from utils import create_adj_s, create_adj_t
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import datetime

class Choosing_steps(nn.Module):
    def __init__(self, file_train, file_valid, file_test, file_label, gru_dim, data_name,
                 time_dim, gru_input_dim, timewindows, sensor_dim, steps=1, seq=1, batch_size=None,
                 num_features_nonzero=None, dropout=0., is_sparse_inputs=False, bias=False,
                 activation = F.relu, featureless=False, BiGru_epochs=1, MSST_GCN_epochs=1, batchs=32):
        '''
        gru_dim:bigru隐变量维度
        time_dim:输入数据的时间维度
        sensor_dim:输入数据的传感器维度
        batch:控制BiGru的训练batch
        GCN的参数(这些参数一般并不需要调整)：
        seq, batch_size, num_features_nonzero,
        dropout, is_sparse_inputs, bias,
        activation, featureless
        steps:
        1:训练BIGRU模型
        2:加载BIGRU模型处理train_data,valid_data和test_data，生成隐变量H
        3:加载train_data，valid_data和test_data的隐变量H，生成相应的邻接矩阵
        4:训练MSST_GCN模型
        5:加载MSST_GCN模型处理测试数据
        注意：这里并不提供反卷积参数的修改，如果需要修改，请到MSST_GCN中进行修改
        '''
        super(Choosing_steps, self).__init__()

        self.steps = steps
        self.file_train = file_train
        self.file_valid = file_valid
        self.file_test = file_test
        self.file_label = file_label
        self.z_dim = gru_dim
        self.data_name = data_name
        self.data_dim = time_dim
        self.gru_input_dim = gru_input_dim
        self.timewindows = timewindows
        self.BiGru_epochs = BiGru_epochs
        self.MSST_GCN_epochs = MSST_GCN_epochs
        self.batch_size = batch_size
        self.output_dim = sensor_dim
        self.seq = seq
        self.batchs = batchs
        self.dropout = dropout
        self.num_features_nonzero = num_features_nonzero
        self.is_sparse_inputs = is_sparse_inputs
        self.bias = bias
        self.activation = activation
        self.featureless = featureless

    def forward(self):

        if(self.steps == 1):

            print("-----Processing 1-----")

            self.BIGRU = BIGRUTrainer(self.data_dim, self.gru_input_dim, self.z_dim, self.output_dim, self.data_name, self.file_train, self.file_valid,
                self.timewindows, self.BiGru_epochs, self.batchs)
            self.BIGRU.train()
            return 1

        elif(self.steps == 2):

            print("-----Processing 2-----")
            begin_time = datetime.datetime.now()
            self.BIGRU = BIGRUTrainer(self.data_dim, self.gru_input_dim, self.z_dim, self.output_dim, self.data_name, self.file_train, self.file_valid,
                self.timewindows, self.BiGru_epochs, self.batchs)
            nums_train = self.BIGRU.predict(self.file_train, mode='train')
            nums_valid = self.BIGRU.predict(self.file_valid, mode='valid')
            nums_test = self.BIGRU.predict(self.file_test)
            end_time = datetime.datetime.now()
            print("Second of BiGRU generating h:", (end_time - begin_time).seconds)
            print("Micro of BiGRU generating h:", (end_time - begin_time).microseconds)
            with open('experiment_data/MSST_GCN.txt', 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                f.write("timestamp: {0}\n".format(timestamp))
                f.write("数据集: {}\n".format(self.data_name))
                f.write("BiGru 处理的训练集节点数: {0}\n".format(nums_train))
                f.write("BiGru 处理的验证集节点数: {0}\n".format(nums_valid))
                f.write("BiGru 处理的测试集节点数: {0}\n".format(nums_test))
                f.write("BiGru_second: {0}\n".format((end_time - begin_time).seconds))
                f.write("BiGru_micro: {0}\n".format((end_time - begin_time).microseconds))
            return 2

        elif(self.steps == 3):

            print("-----Processing 3-----")

            begin_time = datetime.datetime.now()
            train_path = '../save/BIGRU/' + self.data_name + '/H_train.txt'
            valid_path = '../save/BIGRU/' + self.data_name + '/H_valid.txt'
            test_path = '../save/BIGRU/' + self.data_name + '/H_test.txt'

            train_data = np.loadtxt(train_path, delimiter=',')
            valid_data = np.loadtxt(valid_path, delimiter=',')
            test_data = np.loadtxt(test_path, delimiter=',')

            a = 0.8
            b = 0.8

            print("-----Processing train_adj_s-----")
            train_adj_s = create_adj_s(train_data, a)
            print("-----Processing valid_adj_s-----")
            valid_adj_s = create_adj_s(valid_data, a)
            print("-----Processing test_adj_s-----")
            test_adj_s = create_adj_s(test_data, a)
            print("-----Processing train_adj_t-----")
            train_adj_t = create_adj_t(train_data, b)
            print("-----Processing valid_adj_t-----")
            valid_adj_t = create_adj_t(valid_data, b)
            print("-----Processing test_adj_t-----")
            test_adj_t = create_adj_t(test_data, b)

            train_path_s = '../save/BIGRU/' + self.data_name + '/H_train_s.txt'
            valid_path_s = '../save/BIGRU/' + self.data_name + '/H_valid_s.txt'
            test_path_s = '../save/BIGRU/' + self.data_name + '/H_test_s.txt'
            train_path_t = '../save/BIGRU/' + self.data_name + '/H_train_t.txt'
            valid_path_t = '../save/BIGRU/' + self.data_name + '/H_valid_t.txt'
            test_path_t = '../save/BIGRU/' + self.data_name + '/H_test_t.txt'

            np.savetxt(train_path_s, train_adj_s, fmt='%1.6f', delimiter=',')
            np.savetxt(train_path_t, train_adj_t, fmt='%1.6f', delimiter=',')
            np.savetxt(valid_path_s, valid_adj_s, fmt='%1.6f', delimiter=',')
            np.savetxt(valid_path_t, valid_adj_t, fmt='%1.6f', delimiter=',')
            np.savetxt(test_path_s, test_adj_s, fmt='%1.6f', delimiter=',')
            np.savetxt(test_path_t, test_adj_t, fmt='%1.6f', delimiter=',')
            end_time = datetime.datetime.now()
            print("Second of BiGRU generating adj:", (end_time - begin_time).seconds)
            print("Micro of BiGRU generating adj:", (end_time - begin_time).microseconds)
            with open('experiment_data/MSST_GCN.txt', 'a') as f:
                f.write("BiGru_adj_second: {0}\n".format((end_time - begin_time).seconds))
                f.write("BiGru_adj_micro: {0}\n".format((end_time - begin_time).microseconds))
            return 3

        elif(self.steps == 4):

            print("-----Processing 4-----")

            train_path = '../save/BIGRU/' + self.data_name + '/H_train.txt'
            valid_path = '../save/BIGRU/' + self.data_name + '/H_valid.txt'
            test_path = '../save/BIGRU/' + self.data_name + '/H_test.txt'

            train_path_s = '../save/BIGRU/' + self.data_name + '/H_train_s.txt'
            valid_path_s = '../save/BIGRU/' + self.data_name + '/H_valid_s.txt'
            test_path_s = '../save/BIGRU/' + self.data_name + '/H_test_s.txt'
            train_path_t = '../save/BIGRU/' + self.data_name + '/H_train_t.txt'
            valid_path_t = '../save/BIGRU/' + self.data_name + '/H_valid_t.txt'
            test_path_t = '../save/BIGRU/' + self.data_name + '/H_test_t.txt'

            self.MSST_GCN = MSST_GCNTrainer(self.data_dim//self.timewindows * 2, self.output_dim, self.data_name, train_path,
                                            valid_path, train_path_s, train_path_t, valid_path_s, valid_path_t,
                                            test_path, test_path_s, test_path_t, self.file_label,
                                            self.seq, self.MSST_GCN_epochs, self.batchs, self.batch_size, self.num_features_nonzero,
                                            self.dropout, self.is_sparse_inputs, self.bias, self.activation, self.featureless)
            self.MSST_GCN.train()
            return 4

        elif(self.steps == 5):

            print("-----Processing 5-----")

            train_path = '../save/BIGRU/' + self.data_name + '/H_train.txt'
            valid_path = '../save/BIGRU/' + self.data_name + '/H_valid.txt'
            test_path = '../save/BIGRU/' + self.data_name + '/H_test.txt'

            train_path_s = '../save/BIGRU/' + self.data_name + '/H_train_s.txt'
            valid_path_s = '../save/BIGRU/' + self.data_name + '/H_valid_s.txt'
            test_path_s = '../save/BIGRU/' + self.data_name + '/H_test_s.txt'
            train_path_t = '../save/BIGRU/' + self.data_name + '/H_train_t.txt'
            valid_path_t = '../save/BIGRU/' + self.data_name + '/H_valid_t.txt'
            test_path_t = '../save/BIGRU/' + self.data_name + '/H_test_t.txt'

            self.MSST_GCN = MSST_GCNTrainer(self.data_dim//self.timewindows * 2, self.output_dim, self.data_name, train_path,
                                            valid_path, train_path_s, train_path_t, valid_path_s, valid_path_t,
                                            test_path, test_path_s, test_path_t, self.file_label,
                                            self.seq, self.MSST_GCN_epochs, self.batchs, self.batch_size, self.num_features_nonzero,
                                            self.dropout, self.is_sparse_inputs, self.bias, self.activation, self.featureless)
            begin_time = datetime.datetime.now()
            pre_best, acc_best, recall_best, f1_best, threshold_best, points_nums = self.MSST_GCN.evaluate()
            end_time = datetime.datetime.now()
            with open('experiment_data/MSST_GCN.txt', 'a') as f:
                f.write("MSST_GCN_second: {0}\n".format((end_time - begin_time).seconds))
                f.write("MSST_GCN_micro: {0}\n".format((end_time - begin_time).microseconds))
                f.write("Accuracy: {0:.2%}".format(acc_best))
                f.write("Precision: {0:.2%}".format(pre_best))
                f.write("Recall: {0:.2%}".format(recall_best))
                f.write("F1-Score: {0:.2%}\n".format(f1_best))
                f.write("threshold: {0:.2%}\n".format(threshold_best))
                f.write("points_nums: {0}\n".format(points_nums))
            print("pre_best:", pre_best)
            print("acc_best:", acc_best)
            print("recall_best:", recall_best)
            print("f1_best:", f1_best)
            print("threshold_best:", threshold_best)
            print("MSST_GCN_second:", (end_time - begin_time).seconds)
            print("MSST_GCN_micro:", (end_time - begin_time).microseconds)
            print("points_nums: {0}\n".format(points_nums))
            return 5