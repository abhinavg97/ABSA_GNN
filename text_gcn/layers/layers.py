import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class MatrixUpdation(Module):

    def __init__(self, n, d, bias=True):
        super(MatrixUpdation, self).__init__()
        self.n = n
        self.d = d
        self.weight = Parameter(torch.FloatTensor(d, n))
        self.S = Parameter(torch.FloatTensor(n, n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(n))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.S.data)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        S_prime = torch.mul(D_prime, self.S)
        A_prime = torch.mul(S_prime, A)
        X_prime = torch.matmul(A_prime, X)

        X = torch.matmul(X_prime, self.W)

        if self.bias is not None:
            return X + self.bias
        else:
            return X

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
