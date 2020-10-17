import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class MatrixUpdation(Module):

    def __init__(self, n, d, out_dim=2, bias=True):
        super(MatrixUpdation, self).__init__()
        self.n = n
        self.d = d
        self.weight = Parameter(torch.FloatTensor(d, out_dim))
        self.S = Parameter(torch.FloatTensor(n+d, n+d))
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

    def forward(self, A, D_prime, X):
        S_prime = torch.mul(D_prime, self.S)
        A_prime = torch.mul(S_prime, A)
        X_prime = torch.matmul(A_prime, X)

        X = torch.matmul(X_prime, self.weight)

        if self.bias is not None:
            return X + self.bias
        else:
            return X

    def get_dropout_matrix(self, shape_A, dr=0.1):
        """
        updating adjacency matrix according to the logic given in the paper
        Dropout logic is as given here: https://arxiv.org/pdf/1207.0580.pdf
        """

        # document_size is number of documents in the Adj matrix
        # shape of D is document_size x document_size

        D = torch.ones(self.d, self.d)
        dropout = torch.nn.Dropout(p=dr, inplace=False)
        D = dropout(D)

        # D_prime is dropout matrix applied to Adjacency matrix
        # shape of D_prime is same as that of Adjacency matrix

        # dropout = torch.nn.Dropout(p=0.2, inplace=False)
        D_prime = torch.ones(shape_A)

        # D_prime has first dxd elements from D with a higher dropout probability
        # The rest of the elements have a lower dropout probability
        for i in range(self.d):
            for j in range(self.d):
                D_prime[i, j] = D[i, j]

        return D_prime


if __name__ == "__main__":

    n = 5
    d = 2

    m = n + d

    A = torch.randn(m, m)

    dim = 3
    X = torch.randn(m, dim)

    mat_test = MatrixUpdation(n, d, out_dim=2)

    D_prime = mat_test.get_dropout_matrix(A.shape, dr=0.2)

    X_prime = mat_test(A, D_prime, X)

# TODO write backward here, and check if S is getting updated
