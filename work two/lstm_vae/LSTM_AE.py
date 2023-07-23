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
from data_utils import mmd
from data_utils import weights_init, plot_data, read_yahoo
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data_utils import CustomDataset


class Predictor(nn.Module):
    def __init__(self, x_dim, z_dim):
        '''
        :param data_dim: 原始数据的维度
        :param latent_dim: 隐变量维度
        '''
        super(Predictor, self).__init__()
        gru_dim = 100
        # Encoder
        self.encoder1 = nn.GRU(input_size=x_dim, hidden_size=gru_dim, num_layers=1, bias=False, batch_first=True)
        self.encoder2 = nn.InstanceNorm1d(num_features=gru_dim)
        self.encoder3 = nn.Linear(in_features=gru_dim, out_features=z_dim)

        # Decoder
        self.decoder1 = nn.GRU(input_size=z_dim, hidden_size=gru_dim, num_layers=1, bias=False, batch_first=True)
        self.decoder2 = nn.InstanceNorm1d(num_features=gru_dim)
        self.decoder3 = nn.Linear(in_features=gru_dim, out_features=x_dim)

    def forward(self, x):
        x, _ = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x, _ = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x


class PredictorTrainer:
    def __init__(self, data_name, data_root, predictor_path):
        self.data_name = data_name
        self.data_root = data_root
        self.predictor_path = predictor_path
        self.batch_size = 64
        self.time_windows = 10
        self.data_dim = 1  # 数据维度
        self.z_dim = 20  # 隐变量维度
        self.ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
        self.device = torch.device(
            "cuda:3" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
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
        train_data = CustomDataset(data_name=self.data_name, file_path=self.data_root,
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
        mse_loss = nn.MSELoss().cuda(device=self.device)
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

                loss = mse_loss(predict_data, data)
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
        with open('../experiment_data/Evaluate.txt', 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            f.write("timestamp: {0}\n".format(timestamp))
            f.write("数据集: {}\n".format(self.data_root))
            f.write('模型:{}\n'.format(self.predictor_path))
            f.write("Accuracy: {0:.2%}".format(acc))
            f.write("Precision: {0:.2%}".format(pre))
            f.write("Recall: {0:.2%}".format(recall))
            f.write("F1-Score: {0:.2%}\n".format(f1))
            f.write("threshold: {0:.2%}\n".format(threshold))
        writer.close()

    def evaluate(self):
        scores = self.predict()
        if self.data_name == 'Yahoo':
            _, labels = read_yahoo(self.data_root)
        elif self.data_name == 'KPI':
            dataframe = pd.HDFStore(self.data_root).get('data')
            labels = dataframe['label'].values[:10000:2]
        elif self.data_name == 'SMAP':
            labels = pickle.load(open('../data/NASA/SMAP/SMAP_test_label.pkl', 'rb'), encoding='utf-8')
        elif self.data_name == 'MSL':
            labels = pickle.load(open('../data/NASA/MSL/MSL_test_label.pkl', 'rb'), encoding='utf-8')
        else:
            labels = []
        labels = labels[self.time_windows - 1:]
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
            if self.data_name == 'SMAP':
                file_path  = '../data/NASA/SMAP/SMAP_test.pkl'
            elif self.data_name == 'MSL':
                file_path = '../data/NASA/MSL/MSL_test.pkl'
            elif self.data_name == 'Yahoo':
                file_path = self.data_root
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

data_name = 'Yahoo'
for i in range(1, 2):
    data_root = "../data/Yahoo/A1Benchmark/real_" + str(i) + '.csv'  # Root directory for dataset
    predictor_path = '../model_parameter/Lstm_AE_model_noreparameter_Yahoo_' + str(i) + '.pt'

    # train model
    model = PredictorTrainer(data_name=data_name, data_root=data_root, predictor_path=predictor_path)
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
