import torch
import torch.nn.functional as F

from dgl import mean_nodes
from dgl.nn.pytorch.conv import GATConv

import pytorch_lightning as pl


class GAT_Graph_Classifier(pl.LightningModule):
    """
    GAT model class: This is where the learning happens
    The boilerplate for learning is abstracted away by Lightning
    """
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(GAT_Graph_Classifier, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.classify = torch.nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, g, emb=None):
        if emb is None:
            # Use node degree as the initial node feature.
            # For undirected graphs, the in-degree is the
            # same as the out_degree.
            emb = g.ndata['emb']

        # Perform graph convolution and activation function.
        emb = F.relu(self.conv1(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        emb = F.relu(self.conv2(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        g.ndata['emb'] = emb

        # Calculate graph representation by averaging all node representations.
        hg = mean_nodes(g, 'emb')
        return self.classify(hg)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
