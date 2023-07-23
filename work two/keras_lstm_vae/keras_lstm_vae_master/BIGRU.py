import torch.nn as nn
from data_util import weights_init
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from Data_processing import CustomDataset, PredictorDataset
from sklearn.preprocessing import MinMaxScaler

class BIGRU(nn.Module):
    def __init__(self, x_dim, gru_dim):
        '''
        :param data_dim: 原始数据的维度
        :param latent_dim: 隐变量维度
        '''
        super(BIGRU, self).__init__()
        # Encoder
        self.x_dim = x_dim
        self.encoder1 = nn.GRU(input_size=x_dim, hidden_size=gru_dim, num_layers=1,
                               bias=False, batch_first=True, bidirectional=True)
        self.encoder2 = nn.InstanceNorm1d(num_features=gru_dim)
        self.encoder3 = nn.Linear(in_features=2 * gru_dim, out_features=x_dim)

    def loss_function(self, inputs, recons):

        recon_criterion = nn.MSELoss(reduction='sum')
        batch_size = recons.size()[0]
        loss = recon_criterion(inputs, recons)
        loss = loss/float(batch_size)

        return loss

    def forward(self, input_0):

        output_1, hn = self.encoder1(input_0)
        output_2 = self.encoder2(output_1)
        output_3 = self.encoder3(output_2)
        # x = self.decoder_input(z)
        return output_3, hn

class BIGRUTrainer(nn.Module):
    def __init__(self, data_dim, gru_input_dim, gru_dim, sensor_dim, data_name, file_train, file_valid,
                  timewindows, epochs=1, batch_size=32):
        '''
        :param data_dim: 原始数据的维度
        :param gru_dim: 隐变量维度
        '''
        super(BIGRUTrainer, self).__init__()

        # Encoder
        self.z_dim = gru_dim
        self.gru_input_dim = gru_input_dim
        self.sensor_dim = sensor_dim
        self.data_name = data_name
        self.file_train = file_train
        self.file_valid = file_valid
        self.data_dim = data_dim
        self.timewindows = timewindows
        self.epochs = epochs
        self.batch_size = batch_size
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
        self.BIGRU = BIGRU(x_dim=self.data_dim, gru_dim=self.z_dim, sensor_dim=self.sensor_dim).to(self.device)
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.BIGRU = nn.DataParallel(self.BIGRU, list(range(self.ngpu)))
        self.BIGRU.apply(weights_init)
        print(self.BIGRU)

    def train(self):

        save_path = '../save/BIGRU/' + self.data_name
        self.mkdir(save_path)

        train_data = CustomDataset(self.data_name,
                                   self.file_train, self.timewindows, self.data_dim)

        # train_data = TensorDataset(x_t, y_t)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        valid_data = CustomDataset(self.data_name,
                                   self.file_valid, self.timewindows, self.data_dim)

        # train_data = TensorDataset(x_t, y_t)
        valid_loader = DataLoader(dataset=valid_data,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        print(f"Prepared!")

        # model training
        optimizer = optim.Adam(self.BIGRU.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)

        min_valid_loss = float("inf")

        for epoch in range(self.epochs):

            # training
            self.BIGRU.train()

            train_loss = 0.0
            train_cnt = 0

            for index, train_x in enumerate(train_loader, 0):
                #train_x = train_x.cuda(device=self.device)
                train_x = train_x.float()
                train_y = train_x.float()
                optimizer.zero_grad()

                # forward + loss + backward + optimize
                x_recons, H = self.BIGRU(train_x)
                loss_dict = self.BIGRU.loss_function(train_y, x_recons)
                train_loss += loss_dict.item()

                train_cnt += 1
                loss_dict.backward()
                optimizer.step()
                # scheduler.step()

            # validation
            self.BIGRU.eval()
            valid_loss = 0.0
            valid_cnt = 0
            for index, valid_x in enumerate(valid_loader, 0):
                # forward + loss
                #valid_x = valid_x.cuda(device=self.device)
                valid_x = valid_x.float()
                valid_y = valid_x.float()
                valid_x_recons, valid_H = self.BIGRU(valid_x)

                valid_loss_dict = self.BIGRU.loss_function(valid_y, valid_x_recons)

                valid_loss += valid_loss_dict.item()
                # valid_loss += loss
                valid_cnt += 1

            valid_loss = valid_loss / valid_cnt
            # writer.add_scalar('validation loss', valid_loss, epoch)

            print(f"[{epoch}/{self.epochs}]", f"valid_loss:{valid_loss:.2f}", f"train_loss:{(train_loss / train_cnt):.2f}")

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                model_path = os.path.join(save_path, f'_hdim{self.z_dim}.pt')

                torch.save(self.BIGRU.state_dict(), model_path)

        print("-> Training Finished! <-")
        print(f"-> Model is saved in {save_path} <-")

    def predict(self, file, mode="test"):

        save_path = '../save/BIGRU/' + self.data_name
        model_path = os.path.join(save_path, f'_hdim{self.z_dim}.pt')
        scaler = MinMaxScaler(feature_range=(0, 1))
        normal_data = np.loadtxt(file, delimiter=',')
        scaler.fit(normal_data.reshape(-1, self.data_dim))
        # 用正常数据的最大值最小值做归一化
        normalized_data = scaler.transform(normal_data.reshape(-1, self.data_dim))
        torch_data = torch.FloatTensor(normalized_data)
        inout_seq = []
        self.data_length = len(torch_data)
        # 左闭右开
        # for i in range(0, (self.data_length - self.timewindows) // 1 + 1, 1):
        for i in range(0, self.data_length // self.timewindows):
            data_seq = torch_data[i:i + self.timewindows]
            inout_seq.append(data_seq)
        inout_seq = torch.tensor([item.cpu().detach().numpy() for item in inout_seq]).cuda()

        net = self.BIGRU

        net.load_state_dict(torch.load(model_path))
        net.eval()

        # forward + loss
        test_x_recons, H = net(inout_seq.cpu())
        points_nums = 1
        for i in inout_seq.shape:
            points_nums = i * points_nums

        H = H.view((-1, H.shape[-1]))
        np.savetxt(save_path + '/H_' + mode + ".txt", H.detach().numpy(), fmt='%1.6f', delimiter=',')

        return points_nums

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
    data = np.loadtxt(train_path, delimiter=',')
    test_data = np.loadtxt(train_path, delimiter=',')
    timesteps = 10
    z_train = []
    for i in range(len(data) - timesteps - 2):
        x = data[i:(i + timesteps), :]
        z_train.append(x)
    z_train = np.array(z_train)
    y_train = []
    for i in range(1, len(data) - timesteps - 1):
        x = data[i:(i + timesteps), :]
        y_train.append(x)
    y_train = np.array(y_train)
    z_test = []
    for i in range(len(test_data) - timesteps - 1):
        x = test_data[i:(i + timesteps), :]
        z_test.append(x)
    z_test = np.array(z_test)
    model = BIGRUTrainer(38, 38)
    model.train(z_train, y_train)
    preds = model.predict(z_test)
    plt.plot(z_train, label='data')
    plt.plot(preds, label='predict')
    plt.legend()
    plt.show()
# x = torch.randn(32, 10, 64)
# z, H = BIGRU(64, 32)(x)
# print(z.shape)
# print(H.shape)

