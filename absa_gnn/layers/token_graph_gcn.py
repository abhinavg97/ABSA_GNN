import torch
from dgl.nn.pytorch.conv import GraphConv


class Token_Graph_GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Token_Graph_GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, g, emb):
        if emb is None:
            emb = g.ndata['emb']
        emb = self.conv1(g, emb)
        emb = torch.relu(emb)
        emb = self.conv2(g, emb)
        return emb
