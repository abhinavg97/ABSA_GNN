# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : GCN with EdgeDrop on adjacency matrix with learnable update
__description__ :
__project__     :
__classes__     :
__variables__   :
__methods__     :
__author__      : Samujjwal, Abhinav
__version__     : ":  "
__date__        : "30/10/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pytorch_lightning as pl


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ dot product between 2 tensors for dense and sparse format.

    :param x:
    :param y:
    :return:
    """
    if x.is_sparse:
        res = torch.spmm(x, y)
    else:
        res = torch.matmul(x, y)
    return res


class GCN_DropEdgeLearn(pl.LightningModule):

    def __init__(self, w, d, emb_dim, out_dim=2, bias=True):
        super(GCN_DropEdgeLearn, self).__init__()
        self.w = w
        self.d = d
        self.weight = Parameter(torch.FloatTensor(emb_dim, out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.S = Parameter(torch.FloatTensor(w + d, w + d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.S.data)

    def forward(self, A, X):
        torch.autograd.set_detect_anomaly(True)
        # S_prime = torch.mul(D_prime, self.S)
        A_prime = torch.mul(self.S, A)

        # X_prime = torch.matmul(A_prime, X)
        support = torch.mm(X, self.weight)

        # X = torch.matmul(X_prime, self.weight)
        X = dot(A_prime, support)

        if self.bias is not None:
            return X + self.bias
        else:
            return X

    def apply_targeted_dropout(self, targeted_drop=0.1):
        """ Applies dropout to the doc-doc portion of adjacency matrix.

        """
        D = torch.ones(self.d, self.d)
        targeted_dropout = torch.nn.Dropout(p=targeted_drop, inplace=False)
        D = targeted_dropout(D)

        for i in range(self.d):
            for j in range(self.d):
                self.S[i, j] *= D[i, j]

    # def get_targeted_dropout_matrix(self, shape_A, targeted_drop=0.1):
    #     """
    #     updating adjacency matrix according to the logic given in the paper
    #     Dropout logic is as given here: https://arxiv.org/pdf/1207.0580.pdf
    #     """
    #
    #     # self.d is number of documents in the Adj matrix
    #     # shape of D is document_size x document_size
    #
    #     D = torch.ones(self.d, self.d)
    #     targeted_dropout = torch.nn.Dropout(p=targeted_drop, inplace=False)
    #     D = targeted_dropout(D)
    #
    #     # D_prime is dropout matrix applied to Adjacency matrix
    #     # shape of D_prime is same as that of Adjacency matrix
    #
    #     D_prime = torch.ones(shape_A)
    #
    #     # D_prime has first dxd elements from D with a higher dropout probability
    #     # The rest of the elements have a lower dropout probability
    #     for i in range(self.d):
    #         for j in range(self.d):
    #             D_prime[i, j] = D[i, j]
    #
    #     return D_prime

    def apply_adj_dropout(self, adj_drop=0.2):
        """ Applies dropout to the whole adjacency matrix.

        """
        ## Apply dropout to whole adjacency matrix:
        # TODO: symmetric dropout (for undirected graph)
        adj_dropout = torch.nn.Dropout(p=adj_drop, inplace=False)
        self.S = adj_dropout(self.S)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.d) + ')'\
               + str(self.emb_dim) + ' -> ' + str(self.out_dim)


if __name__ == "__main__":
    epochs = 5
    w = 5
    d = 2
    n = w + d
    emb_dim = 2

    A = torch.randn(n, n)
    X = torch.randn(n, emb_dim)
    target = torch.randint(0, 2, (n, emb_dim)).float()

    mat_test = GCN_DropEdgeLearn(w, d, emb_dim=emb_dim, out_dim=emb_dim)

    trainer = pl.Trainer(max_epochs=epochs)

    trainer.fit(mat_test)
