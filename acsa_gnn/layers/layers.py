import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pytorch_lightning as pl


class MatrixUpdation(pl.LightningModule):

    def __init__(self, n, d, emb_dim, out_dim=2, bias=True):
        super(MatrixUpdation, self).__init__()
        self.n = n
        self.d = d
        self.weight = Parameter(torch.FloatTensor(emb_dim, out_dim))
        self.S = Parameter(torch.FloatTensor(n+d, n+d))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
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

    def shared_step(self, batch):
        pass

    def training_step(self, batch):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def loss_function(self, updated_X, target):
        return F.binary_cross_entropy_with_logits(updated_X, target)


if __name__ == "__main__":

    epochs = 5
    n = 5
    d = 2
    m = n + d
    A = torch.randn(m, m)
    emb_dim = 1
    X = torch.randn(m, emb_dim)
    mat_test = MatrixUpdation(n, d, emb_dim=emb_dim, out_dim=emb_dim)

    train_epoch_losses = []

    optimizer = torch.optim.Adam(mat_test.parameters(), lr=0.001)

    target = torch.randint(0, 2, (m, 1)).float()
    # target = torch.empty(m, dtype=torch.long).random_(emb_dim)
    for epoch in range(epochs):

        epoch_loss = 0
        mat_test.train()

        D_prime = mat_test.get_dropout_matrix(A.shape, dr=0.2+(epoch/10))
        updated_X = mat_test(A, D_prime, X)

        loss = F.binary_cross_entropy_with_logits(updated_X, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        train_epoch_losses.append(epoch_loss)
        X = updated_X
