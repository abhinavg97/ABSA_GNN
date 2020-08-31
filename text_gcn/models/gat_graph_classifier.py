import torch
from dgl.nn.pytorch.conv import GATConv
import torch.nn.functional as F
from dgl import mean_nodes


class GAT_Graph_Classifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(GAT_Graph_Classifier, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.classify = torch.nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, g, h=None):
        if h is None:
            # Use node degree as the initial node feature.
            # For undirected graphs, the in-degree is the
            # same as the out_degree.
            h = g.in_degrees().view(-1, 1).float()

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = h.view(-1, h.size(1) * h.size(2)).float()
        h = F.relu(self.conv2(g, h))
        h = h.view(-1, h.size(1) * h.size(2)).float()
        g.ndata['h'] = h

        # Calculate graph representation by averaging all node representations.
        hg = mean_nodes(g, 'h')
        return self.classify(hg)
