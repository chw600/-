from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from BIGRU import BIGRU


# seed for reproduce
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# make dir
def mkdir(path):
    path = path.strip()
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def training(batch,
             in_channels=2,
             latent_dim=128,
             epochs=200,
             gpu_index=None):
    # ================================= GPU ==========================================
    # gpu config
    if gpu_index is None:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_index}')
    else:
        raise Exception('Input correct #GPU !')

    # ================================= data ==========================================
    # data configuration
    BATCH_SIZE = batch
    # model saving config
    save_path = '../save/BIGRU/'
    mkdir(save_path)

    # data loading
    # data_dict = torch.load(f'../data/processed/{dataset}.pt', map_location='cuda:0')


    train_set = TensorDataset(data_dict['x_train'], data_dict['y_train'])
    train_loader = DataLoader(dataset=train_set,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_set = TensorDataset(data_dict['x_valid'], data_dict['y_valid'])
    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    print("Data Prepared !")

    # ==================== model_name selecting ============================

    net = BIGRU(x_dim=in_channels, z_dim=latent_dim)

    # ================================ model training ============================
    net = net.to(device)

    # tensorboard config
    # writer = SummaryWriter('logs/' + f'{dataset}/' + model + '_ratio' + str(mis_ratio))

    print(f"Prepared!")

    # model training
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)

    min_valid_loss = float("inf")

    for epoch in range(epochs):

        # training
        net.train()

        train_loss = 0.0
        recon_loss = 0.0
        train_cnt = 0

        for index, (x_batch, _) in enumerate(train_loader, 0):
            x = x_batch[:, 0].to(device)
            optimizer.zero_grad()

            # forward + loss + backward + optimize
            x_recons, H = net(x)

            loss_dict = net.loss_function(x, x_recons)
            train_loss += loss_dict.item()

            train_cnt += 1
            loss_dict.backward()
            optimizer.step()
            # scheduler.step()

        # writer.add_scalar('training loss', train_loss / train_cnt, epoch)

        # validation
        net.eval()
        valid_loss = 0.0
        valid_cnt = 0
        for index, (valid_x, _) in enumerate(valid_loader, 0):
            x = valid_x[:, 0].to(device)
            mask = valid_x[:, 1].to(device)
            # forward + loss
            valid_x_recons, valid_H = net(x)

            valid_loss_dict = net.loss_function(x, valid_x_recons)

            valid_loss += valid_loss_dict.item()
            # valid_loss += loss
            valid_cnt += 1

        valid_loss = valid_loss / valid_cnt
        # writer.add_scalar('validation loss', valid_loss, epoch)

        print(f"[{epoch}/{epochs}]", f"valid_loss:{valid_loss:.2f}", f"train_loss:{(train_loss / train_cnt):.2f}")

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            model_path = os.path.join(save_path, f'_hdim{latent_dim}.pt')

            torch.save(net.state_dict(), model_path)

    print("-> Training Finished! <-")
    print(f"-> Model is saved in {save_path} <-")


if __name__ == "__main__":
    seed_everything(seed=42)


    training(batch=32,in_channels=2,
                 latent_dim=128,
                 epochs=1500,
                 warmup=500,
                 gated=True,
                 gpu_index=0)
