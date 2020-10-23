import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pytorch_lightning as pl


class MatrixUpdation(pl.LightningModule):

    def __init__(self, w, d, X, A, target, emb_dim, out_dim=2, bias=True):
        super(MatrixUpdation, self).__init__()
        self.w = w
        self.d = d
        self.X = X
        self.A = A
        self.target = target
        self.weight = Parameter(torch.FloatTensor(emb_dim, out_dim))
        self.S = Parameter(torch.FloatTensor(w+d, w+d))
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
        torch.autograd.set_detect_anomaly(True)
        S_prime = torch.mul(D_prime, self.S)
        A_prime = torch.mul(S_prime, A)
        X_prime = torch.matmul(A_prime, X)

        X = torch.matmul(X_prime, self.weight)

        if self.bias is not None:
            return X + self.bias
        else:
            return X

    def backward(self, loss, optimizer, optimizer_idx):
        # do a custom way of backward
        loss.backward(retain_graph=True)

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

    def train_dataloader(self):
        return [self.A]

    def training_step(self, batch, batch_idx):
        D_prime = self.get_dropout_matrix(self.A.shape, dr=0.2+(self.current_epoch/10))
        updated_X = self(self.A, D_prime, self.X)
        batch_loss = self.loss_function(updated_X, self.target)
        # self.X = updated_X
        return {'loss': batch_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def loss_function(self, updated_X, target):
        return F.binary_cross_entropy_with_logits(updated_X, target)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == "__main__":

    epochs = 5
    w = 5
    d = 2
    n = w + d
    emb_dim = 1

    A = torch.randn(n, n)
    X = torch.randn(n, emb_dim)
    target = torch.randint(0, 2, (n, 1)).float()

    mat_test = MatrixUpdation(w, d, X, A, target, emb_dim=emb_dim, out_dim=emb_dim)

    trainer = pl.Trainer(max_epochs=epochs)

    trainer.fit(mat_test)
