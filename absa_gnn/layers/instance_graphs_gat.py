import torch
import torch.nn.functional as F

from dgl.nn.pytorch.conv import GATConv

import pytorch_lightning as pl


class Instance_Graphs_GAT(pl.LightningModule):
    """
    Layer
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, out_dim: int) -> None:
        super(Instance_Graphs_GAT, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, emb: torch.Tensor = None) -> torch.Tensor:
        if emb is None:
            emb = g.ndata['emb']

        # Perform graph convolution and activation function.
        emb = F.relu(self.conv1(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        emb = F.relu(self.conv2(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        # g.ndata['emb'] = emb
        return emb

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
