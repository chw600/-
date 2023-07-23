import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import math
import datetime
import os
import pickle

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc, \
    average_precision_score
from data_util import mmd
from data_util import weights_init, plot_data, read_yahoo
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data_util import CustomDataset

class Predictor(nn.Module):
    def __init__(self, x_dim, z_dim):
        '''
        :param data_dim: 原始数据的维度
        :param latent_dim: 隐变量维度
        '''
        super(Predictor, self).__init__()
        gru_dim = 100
        # Encoder
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encoder1 = nn.GRU(input_size=x_dim, hidden_size=gru_dim, num_layers=1, bias=False, batch_first=True)
        self.encoder2 = nn.InstanceNorm1d(num_features=gru_dim)
        # self.encoder3 = nn.Linear(in_features=gru_dim, out_features=z_dim)

        # Decoder
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv_3= nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.deconv_1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3)
        self.deconv_3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3)
        self.fc = nn.Linear(in_features=gru_dim, out_features=x_dim)

    def forward(self, x):
        x, _ = self.encoder1(x)
        x = self.encoder2(x)
        # x.shape=(64, 10, 100)
        b, h, w = x.shape
        x = x.reshape(b, 1, h, w)
        # x = self.encoder3(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = x.reshape(b, h, w)
        x = self.fc(x)
        return x

class PredictorTrainer:
    def __init__(self, data_name, data_root, predictor_path, nums=0):
        self.data_name = data_name
        # self.data_root = data_root
        self.predictor_path = predictor_path
        self.batch_size = 64
        self.time_windows = 10
        self.data_dim = 25  # 数据维度
        self.z_dim = 20  # 隐变量维度
        self.ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
        self.nums = nums
        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.build_models()
        print('dataset name:{}'.format(data_name))
        print('time windows length:{}'.format(self.time_windows))

    def build_models(self):
        # 初始化LSTM
        self.predictor = Predictor(x_dim=self.data_dim, z_dim=self.z_dim).to(self.device)
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.predictor = nn.DataParallel(self.predictor, list(range(self.ngpu)))
        self.predictor.apply(weights_init)
        print(self.predictor)

    def train(self):
        # file_path = 'ServerMachineDataset/train/machine-3-'+str(self.nums)+'.txt'
        file_path = 'SMAP/train_selective/T-3.npy'
        train_data = CustomDataset(data_name=self.data_name, file_path=file_path,
                                   time_windows=self.time_windows,
                                   step=1, mode='Train')
        data_loader = DataLoader(dataset=train_data, batch_size=self.batch_size,
                                 shuffle=True)
        log_dir = 'log/'
        writer = SummaryWriter(log_dir)
        self.data_loader = data_loader
        lr = 0.00001
        beta1 = 0.9
        num_epochs = 10
        consume_time_list = []
        # 初始化loss函数及优化器
        optimizer = optim.Adam(self.predictor.parameters(), lr=lr, betas=(beta1, 0.999))
        iteration_number = math.ceil(train_data.__len__() / self.batch_size)
        print('LSTM save path:{}'.format(self.predictor_path))
        print("Starting Training Loop...")

        # For each epoch
        for epoch in range(num_epochs):
            i = 0
            losses_total = 0
            for data in data_loader:
                i += 1
                data = data.cuda(device=self.device)
                # ====== update LSTM ======
                # encoder forward
                self.predictor.zero_grad()
                predict_data = self.predictor(data)

                loss = self.loss(predict_data, data)
                loss.backward()
                optimizer.step()

                # Save Losses for plotting later
                losses_total = losses_total + loss

                '''
                # 保存loss至tensorboard
                writer.add_scalar('encoder_generator_train_lose', loss_eg.item(), epoch * iteration_number + i)
                writer.add_scalar('discriminator_train_lose', loss_d.item(), epoch * iteration_number + i)
                writer.add_scalar('RMSE', RMSE, epoch * iteration_number + i)
                '''

            # Output training stats
            print('[%d/%d]\tLoss: %.4f\t' % (epoch + 1, num_epochs, losses_total / iteration_number))
            self.save_weight()
            starttime = datetime.datetime.now()
            pre, acc, recall, f1, threshold = self.evaluate()
            endtime = datetime.datetime.now()
            print((endtime - starttime).seconds)
            print((endtime - starttime).microseconds)
            consume_time_list.append((endtime - starttime).microseconds)
            print('f1:{}'.format(f1))
        print(np.mean(consume_time_list))
        with open('experiment_data/Evaluate_lstm_cnn_SMAP.txt', 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            f.write("timestamp: {0}\n".format(timestamp))
            f.write("数据集: {}\n".format("SMAP_T_3"))
            f.write('模型:{}\n'.format("LSTM_CNN"))
            f.write("Accuracy: {0:.2%}".format(acc))
            f.write("Precision: {0:.2%}".format(pre))
            f.write("Recall: {0:.2%}".format(recall))
            f.write("F1-Score: {0:.2%}\n".format(f1))
            f.write("threshold: {0:.2%}\n".format(threshold))
            f.write("second: {0}\n".format((endtime - starttime).seconds))
            f.write("micro: {0}\n".format((endtime - starttime).microseconds))
        writer.close()

    def evaluate(self):
        scores = self.predict()
        # test_label_path = "ServerMachineDataset/test_label/machine-3-"+str(self.nums)+".txt"
        test_label_path = "SMAP/label_selective/T-3.npy"
        # labels = np.loadtxt(test_label_path)
        labels = np.load(test_label_path)
        labels = np.reshape(labels, (labels.shape[0]))
        labels = labels[self.time_windows - 1:]
        f1_best = 0
        threshold_best = 0
        pre_best = 0
        acc_best = 0
        recall_best = 0

        print(labels, scores)
        fpr, tpr, ths = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        for threshold in np.arange(0., 1., 0.05):
            tmp_scores = scores.copy()
            tmp_scores[tmp_scores >= threshold] = 1
            tmp_scores[tmp_scores < threshold] = 0
            if self.data_name == "Yahoo":
                f1 = f1_score(labels, tmp_scores)
                if f1 > f1_best:
                    f1_best = f1
                    pre_best = precision_score(labels, tmp_scores)
                    acc_best = accuracy_score(labels, tmp_scores)
                    recall_best = recall_score(labels, tmp_scores)
                    threshold_best = threshold
            else:
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
        return pre_best, acc_best, recall_best, f1_best, threshold_best

    def predict(self, scale=True):
        '''
        用该点的前time_windows-1个点来判断该点是否异常
        :param scale:是否将分数归一化到0~1，默认为True
        :return: 返回每个点的分数
        '''
        with torch.no_grad():
            batch_size = 64
            # file_path = 'ServerMachineDataset/test/machine-3-'+str(self.nums)+'.txt'
            file_path = "SMAP/test_selective/T-3.npy"
            test_data = CustomDataset(data_name=self.data_name, file_path=file_path,
                                      time_windows=self.time_windows, step=1, mode='Test')
            test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

            scores = torch.zeros(size=(test_data.__len__(),), dtype=torch.float32, device=self.device)

            for i, data in enumerate(test_loader):
                data = data.cuda(device=self.device)
                data = data.to(self.device)
                predict = self.predictor(data)

                if self.data_name == 'Yahoo':
                    error = torch.pow((data.view(data.size(0), -1) - predict.view(predict.size(0), -1)), 2)
                    scores[i * batch_size: i * batch_size + error.size(0)] = error[:, -1:].view(-1)
                else:
                    data = data[:, -1:, :]
                    predict = predict[:, -1:, :]
                    error = torch.pow((data.view(data.size(0), -1) - predict.view(predict.size(0), -1)), 2)
                    error = error.mean(axis=1)
                    scores[i * batch_size: i * batch_size + error.size(0)] = error

            # Scale error vector between [0, 1]
            if scale:
                scores = (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))

            scores = scores.cpu().numpy()

            return scores

    def save_weight(self):
        save_dir = "model_parameter"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.predictor, self.predictor_path)

    def load(self):
        self.predictor = torch.load(self.predictor_path)

    def loss(self, recons, inputs):
        recon_criterion = nn.MSELoss(reduction='sum')
        batch_size = recons.size()[0]
        # 重建损失 -> scalar

        recons_loss = recon_criterion(recons, inputs)

        recons_loss = recons_loss / float(batch_size)
        return recons_loss


class MmdLoss(nn.Module):
    def __init__(self, source, target):
        super(MmdLoss, self).__init__()
        self.source = source
        self.target = target

    def forward(self):
        loss = mmd(self.source, self.target)
        return loss


# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

data_name = 'SMD'
# for i in range(1, 12):
#     data_root = "data/SMD/machine_" + str(i) + '.csv'  # Root directory for dataset
#     predictor_path = 'model_parameters/Lstm_VAE_model_noreparameter_SMD_' + str(i) + '.pt'
#
#     # train model
#     model = PredictorTrainer(data_name=data_name, data_root=data_root, predictor_path=predictor_path, nums=i)
#     model.train()
data_root = "data/SMAP/T_3.csv"  # Root directory for dataset
predictor_path = 'model_parameters/Lstm_VAE_model_noreparameter_SMAP_T_3.pt'

# train model
model = PredictorTrainer(data_name=data_name, data_root=data_root, predictor_path=predictor_path, nums=0)
model.train()

# # ====== SMAP dataset ===== #
# data_name = 'SMAP'
# data_root = '../data/NASA/SMAP/SMAP_test.pkl'  # Root directory for dataset
# predictor_path = '../model_parameter/Lstm_AE_model_SMAP.pt'
#
# # train model
# model = PredictorTrainer(data_name=data_name, data_root=data_root, predictor_path=predictor_path)
# model.train()

# ====== MSL dataset ===== #
# data_name = 'MSL'
# data_root = '../data/NASA/MSL/MSL_train.pkl'  # Root directory for dataset
# predictor_path = '../model_parameter/Lstm_AE_model_MSL.pt'
#
# # train model
# model = PredictorTrainer(data_name=data_name, data_root=data_root, predictor_path=predictor_path)
# model.train()
