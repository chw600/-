import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import sparse_dropout

class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim, seq=1,
                 batch_size=None,
                 num_features_nonzero=None,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero
        self.weight = nn.Parameter(torch.randn(seq, input_dim, output_dim))
        self.bias = None
        if batch_size:
            self.weight = nn.Parameter(torch.randn(batch_size, seq, input_dim, output_dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(batch_size, seq, output_dim))
        else:
            if bias:
                self.bias = nn.Parameter(torch.zeros(seq, output_dim))

    def forward(self, x, x_adj):
        # print('inputs:', x.shape)

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        # print("x:", x.shape, "weight:", self.weight.shape, "x_adj:", x_adj.shape)
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.matmul(x, self.weight)
            # print("xw:", xw.shape)
        else:
            xw = self.weight
        # print("xw:", xw.shape)
        out = torch.matmul(x_adj, xw)
        # print(out.shape)

        if self.bias is not None:
            out += self.bias

        return self.activation(out)


