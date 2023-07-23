import torch.nn as nn
import torch
import random
import numpy as np
from GCN import GraphConvolution
import torch.nn.functional as F
from data_util import weights_init
import os
import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from Data_processing import CustomDataset, Dealing_data
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc, \
    average_precision_score

class MSST_GCN(nn.Module):
    def __init__(self, x_dim_s, x_dim_t, seq=1,
                 batch_size=None,
                 num_features_nonzero=None,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        '''
        注意：这里反卷积的卷积核大小使用为 1。
        '''
        super(MSST_GCN, self).__init__()
    #    densing:
        self.x_dim_s = x_dim_s
        self.x_dim_t = x_dim_t
        self.seq = seq
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_features_nonzero = num_features_nonzero
        self.is_sparse_inputs = is_sparse_inputs
        self.bias = bias
        self.activation = activation
        self.featureless = featureless
        self.s_gcn_1 = GraphConvolution(self.x_dim_s, 8, self.seq, self.batch_size,
                                      self.num_features_nonzero, self.dropout, self.is_sparse_inputs,
                                      self.bias, self.activation, self.featureless)
        self.s_gcn_2 = GraphConvolution(8, 4, self.seq, self.batch_size,
                                        self.num_features_nonzero, self.dropout, self.is_sparse_inputs,
                                        self.bias, self.activation, self.featureless)
        self.s_gcn_3 = GraphConvolution(4, self.x_dim_s, self.seq, self.batch_size,
                                        self.num_features_nonzero, self.dropout, self.is_sparse_inputs,
                                        self.bias, self.activation, self.featureless)
        self.t_gcn_1 = GraphConvolution(self.x_dim_t, 8, self.seq, self.batch_size,
                                      self.num_features_nonzero, self.dropout, self.is_sparse_inputs,
                                      self.bias, self.activation, self.featureless)
        self.t_gcn_2 = GraphConvolution(8, 4, self.seq, self.batch_size,
                                        self.num_features_nonzero, self.dropout, self.is_sparse_inputs,
                                        self.bias, self.activation, self.featureless)
        self.t_gcn_3 = GraphConvolution(4, self.x_dim_t, self.seq, self.batch_size,
                                        self.num_features_nonzero, self.dropout, self.is_sparse_inputs,
                                        self.bias, self.activation, self.featureless)
        self.decoder_1 = nn.ConvTranspose2d(in_channels=2, out_channels=8, kernel_size=1)
        self.decoder_2 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=1)
        self.decoder_3 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=1)
        self.fc = nn.Linear(self.x_dim_t, self.x_dim_t)

    def forward(self, x, x_adj_s, x_adj_t):

        x_t = self.t_gcn_1(x, x_adj_t)
        x = torch.transpose(x, 1, 0)
        x_s = self.s_gcn_1(x, x_adj_s)
        x_s = self.s_gcn_2(x_s, x_adj_s)
        x_t = self.t_gcn_2(x_t, x_adj_t)
        x_s = self.s_gcn_3(x_s, x_adj_s)
        x_t = self.t_gcn_3(x_t, x_adj_t)
        x_s = torch.transpose(x_s, 2, 1)
        # print("x_s:", x_s, "x_t:", x_t)
        x_s = x_s.unsqueeze(1)
        x_t = x_t.unsqueeze(1)
        # print("x_s:", x_s.shape, "x_t:", x_t.shape)
        x_fusion = torch.cat((x_s, x_t), 1)
        # print("x_fusion:", x_fusion.shape)
        output = self.decoder_1(x_fusion)
        output = self.decoder_2(output)
        output = self.decoder_3(output)
        _, _, T, K = output.shape
        output = output.view((T, K))
        output = self.fc(output)

        return output

    def loss(self, inputs, recons):

        loss_model = nn.MSELoss(reduction="sum")
        loss = loss_model(inputs, recons)
        batch_size = inputs.shape[0]
        loss = loss / batch_size

        return loss

class MSST_GCNTrainer(nn.Module):
    def __init__(self, x_dim_s, x_dim_t, data_name, file_train, file_valid,
                 file_adj_s, file_adj_t, file_adj_s_val, file_adj_t_val,
                 file_test, file_adj_s_test, file_adj_t_test, file_test_label,
                 seq=1, epochs=1, batchs=32,
                 batch_size=None,
                 num_features_nonzero=None,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        '''
        :param data_dim: 原始数据的维度
        :param latent_dim: 隐变量维度
        '''
        super(MSST_GCNTrainer, self).__init__()

        # Encoder
        self.x_dim_s = x_dim_s
        self.x_dim_t = x_dim_t
        self.seq = seq
        self.data_name = data_name
        self.file_train = file_train
        self.file_valid = file_valid
        self.file_adj_s = file_adj_s
        self.file_adj_t = file_adj_t
        self.file_adj_s_val = file_adj_s_val
        self.file_adj_t_val = file_adj_t_val
        self.file_test = file_test
        self.file_adj_s_test = file_adj_s_test
        self.file_adj_t_test = file_adj_t_test
        self.file_test_label = file_test_label
        self.epochs = epochs
        self.batchs = batchs
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_features_nonzero = num_features_nonzero
        self.is_sparse_inputs = is_sparse_inputs
        self.bias = bias
        self.activation = activation
        self.featureless = featureless
        self.ngpu = 0  # Number of GPUs available. Use 0 for CPU mode.
        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.build_models()

    def mkdir(self, path):
        path = path.strip()
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)

    def build_models(self):
        # 初始化LSTM
        self.MSST_GCN = MSST_GCN(self.x_dim_s, self.x_dim_t, self.seq, self.batch_size,
                                      self.num_features_nonzero, self.dropout, self.is_sparse_inputs,
                                      self.bias, self.activation, self.featureless).to(self.device)
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.MSST_GCN = nn.DataParallel(self.MSST_GCN, list(range(self.ngpu)))
        self.MSST_GCN.apply(weights_init)
        print(self.MSST_GCN)

    def train(self):

        save_path = '../save/MSST_GCN/' + self.data_name
        self.mkdir(save_path)

        train_data = Dealing_data(self.file_train)

        train_adj_s = Dealing_data(self.file_adj_s)

        train_adj_t = Dealing_data(self.file_adj_t)

        valid_data = Dealing_data(self.file_valid)

        valid_adj_s = Dealing_data(self.file_adj_s_val)

        valid_adj_t = Dealing_data(self.file_adj_t_val)

        print(f"Prepared!")

        # model training
        optimizer = optim.Adam(self.MSST_GCN.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)

        min_valid_loss = float("inf")

        for epoch in range(self.epochs):

            # training
            self.MSST_GCN.train()

            train_loss = 0.0
            train_cnt = 0

            # forward + loss + backward + optimize
            recons = self.MSST_GCN(train_data, train_adj_s, train_adj_t)
            # print("recons:", recons.shape, "train_data:", train_data.shape)
            loss_dict = self.MSST_GCN.loss(train_data, recons)
            train_loss += loss_dict.item()

            train_cnt += 1
            optimizer.zero_grad()
            loss_dict.backward()
            optimizer.step()

            # validation
            self.MSST_GCN.eval()
            valid_loss = 0.0
            valid_cnt = 0
            # forward + loss

            valid_recons = self.MSST_GCN(valid_data, valid_adj_s, valid_adj_t)
            valid_loss_dict = self.MSST_GCN.loss(valid_data, valid_recons)

            valid_loss += valid_loss_dict.item()
            # valid_loss += loss
            valid_cnt += 1

            valid_loss = valid_loss / valid_cnt
            # writer.add_scalar('validation loss', valid_loss, epoch)

            print(f"[{epoch}/{self.epochs}]", f"valid_loss:{valid_loss:.2f}", f"train_loss:{(train_loss / train_cnt):.2f}")

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                model_path = os.path.join(save_path, f'_hdim{self.x_dim_t}.pt')

                torch.save(self.MSST_GCN.state_dict(), model_path)

        print("-> Training Finished! <-")
        print(f"-> Model is saved in {save_path} <-")

        print(f"-> Predict <-")

    def predict(self):

        save_path = '../save/MSST_GCN/' + self.data_name
        model_path = os.path.join(save_path, f'_hdim{self.x_dim_t}.pt')

        test_data = Dealing_data(self.file_test)
        test_adj_s = Dealing_data(self.file_adj_s_test)
        test_adj_t = Dealing_data(self.file_adj_t_test)

        net = self.MSST_GCN

        net.load_state_dict(torch.load(model_path))
        net.eval()

        # forward + loss
        test_x_recons = net(test_data, test_adj_s, test_adj_t)

        self.mkdir(save_path+"/new")
        data_path = os.path.join(save_path+"/new", f'_hdim{self.x_dim_t}.txt')
        np.savetxt(data_path, test_x_recons.detach().numpy(), fmt="%1.6f", delimiter=",")

        scores = torch.zeros(size=(test_data.shape[1],), dtype=torch.float32, device=self.device)
        error = torch.pow((test_data.view(test_data.size(0), -1)
                           - test_x_recons.view(test_x_recons.size(0), -1)), 2)
        error = error.mean(axis=0)
        scores = error

        points_nums = 1
        for i in test_data.shape:
            points_nums *= i

        # Scale error vector between [0, 1]
        if(torch.max(scores) != torch.min(scores)):
            scores = (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))
        scores = scores.detach().numpy()

        return scores, points_nums

    def evaluate(self):

        scores, points_nums = self.predict()
        test_label_path = self.file_test_label
        labels = np.loadtxt(test_label_path, delimiter=",")
        f1_best = 0
        threshold_best = 0
        pre_best = 0
        acc_best = 0
        recall_best = 0

        fpr, tpr, ths = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)


        for threshold in np.arange(0., 1., 0.05):
            tmp_scores = scores.copy()
            tmp_scores[tmp_scores >= threshold] = 1
            tmp_scores[tmp_scores < threshold] = 0
            TP = 0
            FN = 0
            FP = 0
            # 计算TP和FN
            index = []
            truth = np.where(labels == 1)[0]
            tuple = (truth[0],)
            for i in range(len(truth) - 1):
                if truth[i] + 1 != truth[i + 1]:
                    tuple = tuple + (truth[i],)
                    index.append(tuple)
                    tuple = (truth[i + 1],)
            for i in index:
                if sum(labels[i[0]:i[1]] * tmp_scores[i[0]:i[1]]) > 0:  # 实际为异常值，且检测到了异常值
                    TP = TP + 1
                else:  # 实际为异常值，但是没有检测到异常值
                    FN = FN + 1
            # 计算FP
            index = []
            labels = 1 - labels
            truth = np.where(labels == 1)[0]
            tuple = (truth[0],)
            for i in range(len(truth) - 1):
                if truth[i] + 1 != truth[i + 1]:
                    tuple = tuple + (truth[i],)
                    index.append(tuple)
                    tuple = (truth[i + 1],)
            for i in index:
                if sum(labels[i[0]:i[1]] * tmp_scores[i[0]:i[1]]) > 0:  # 实际为正常值，检测成了异常值
                    FP = FP + 1
            precision = TP / (TP + FP + 0.00001)
            recall = TP / (TP + FN + 0.00001)
            f1 = 2 * precision * recall / (precision + recall + 0.00001)
            if f1 > f1_best:
                f1_best = f1
                pre_best = precision
                acc_best = accuracy_score(labels, tmp_scores)  # 此处的acc是错误的
                recall_best = recall
                threshold_best = threshold

        auc_prc = average_precision_score(labels, scores)
        return pre_best, acc_best, recall_best, f1_best, threshold_best, points_nums


if __name__ == "__main__":
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # timewindows = 2
    # a = torch.tensor([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
    # b = torch.tensor([30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195])
    # c = b[::-1]
    #
    # x_train = 10 * np.cos(a * np.pi / 180)
    # x_train = np.expand_dims(x_train, 0)
    # x_train = x_train.reshape(1, -1)
    # x_train = np.repeat(x_train, 4, axis=0)
    # x_train = x_train.reshape(-1, 1)
    # x_train = np.expand_dims(x_train, 0)
    # x_train = np.repeat(x_train, 1, axis=0)
    #
    # y_train_1 = 10 * np.cos(b * np.pi / 180)
    # y_train_1 = np.expand_dims(y_train_1, 0)
    # y_train_1 = y_train_1.reshape(1, -1)
    # y_train_1 = np.repeat(y_train_1, 4, axis=0)
    # y_train_1 = y_train_1.reshape(-1, 1)
    # y_train_1 = np.expand_dims(y_train_1, 0)
    # y_train_1 = np.repeat(y_train_1, 1, axis=0)
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

    train_path = 'ServerMachineDataset/train/machine-1-3.txt'
    test_path = 'ServerMachineDataset/test/machine-1-3.txt'
    model = MSST_GCNTrainer(38, 38)
    model.train()
    plt.plot(z_train, label='data')
    plt.plot(preds, label='predict')
    plt.legend()
    plt.show()
# x = torch.randn(32, 10, 64)
# z, H = BIGRU(64, 32)(x)
# print(z.shape)
# print(H.shape)

