import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
from BIGRU import BIGRU
from config import args

from utils import masked_loss, masked_acc

seed = 999
np.random.seed(seed)
torch.random.manual_seed(seed)

# load data

device = torch.device('cpu')
# Batch_size, Sequence/Time, Location, Features
x_train = torch.randn(64, 10, 38, device=device)

net = BIGRU(x_dim=38, z_dim=100)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

net.train()
for epoch in range(args.epochs):

    out = net((x_train))
    out = out[0]

    loss = nn.MSELoss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Train: [{0}/{1}]\t'.format(epoch, args.epochs))
        # 'Loss: {0}\t'.format(loss.item()))

net.eval()

# out = net((feature, support))
# out = out[0]
