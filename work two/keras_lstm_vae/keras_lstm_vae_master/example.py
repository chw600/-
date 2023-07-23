import numpy as np
import matplotlib.pyplot as plt

from data_utils import utils
from lstm_vae import create_lstm_vae

def get_data(if_train=True, nums=1):
    # read data from file
    # data = np.fromfile('sample_data.dat').reshape(419,13)
    if(if_train):
        train_path = 'ServerMachineDataset/train/machine-1-' + str(nums) + '.txt'
        data = np.loadtxt(train_path, delimiter=',')
    else:
        test_path = 'ServerMachineDataset/test/machine-1-' + str(nums) + '.txt'
        data = np.loadtxt(test_path, delimiter=',')
    timesteps = 10
    dataX = []
    for i in range(len(data) - timesteps - 1):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)


if __name__ == "__main__":
    for j in range(3, 4):
        x = get_data(if_train=True, nums=j)
        # x = get_data(if_train=False, nums=j)
        input_dim = x.shape[-1]  # 13
        timesteps = x.shape[1]  # 3
        batch_size = 1

        vae, enc, gen = create_lstm_vae(input_dim,
                                        timesteps=timesteps,
                                        batch_size=batch_size,
                                        intermediate_dim=32,
                                        latent_dim=100,
                                        epsilon_std=1.)

        vae.fit(x, x, epochs=100)

        x = get_data(if_train=False, nums=j)
        preds = vae.predict(x, batch_size=batch_size)

        test_label_path = "ServerMachineDataset/test_label/machine-1-" + str(j) + ".txt"
        test_label = np.loadtxt(test_label_path)
        # scores = torch.zeros(size=(x.__len__(),), dtype=torch.float32)
        # pick a column to plot.
        print("[plotting...]")
        print("x: %s, preds: %s" % (x.shape, preds.shape))
        datatest_label = []
        for i in range(len(test_label) - timesteps - 1):
            y = test_label[i:(i + timesteps)]
            datatest_label.append(y)
        test_label = np.array(datatest_label)
        utils.lstm_evaluate(x, preds, test_label)
        # error = torch.pow((x.view(x.shape(0), -1) - preds.view(preds.size(0), -1)), 2)
        # scores[batch_size: batch_size + error.size(0)] = error[:, -1:].view(-1)
        # plt.plot(x[:, 0, 3], label='data')
        # plt.plot(preds[:, 0, 3], label='predict')
        # plt.legend()
        # plt.show()


