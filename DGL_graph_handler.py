import torch
import networkx as nx
import matplotlib.pyplot as plt
from dgl import batch as g_batch, mean_nodes
from dgl.nn.pytorch.conv import GATConv
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from text_gcn.loaders import GraphDataset


def batch_graphs(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = g_batch(graphs)
    return batched_graph, torch.tensor(labels)


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


def train_graph_classifier(model, data_loader, loss_func, optimizer, epochs=5):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for iter, (graph_batch, label) in enumerate(data_loader):
            prediction = model(graph_batch)
            label = label.type_as(prediction)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)


def test_graph_classifier():
    pass


def graph_multiclass_classification(in_feats=1, hid_feats=4, num_heads=2):

    # Create training and test sets.
    file_path = "/home/abhi/Desktop/gcn/data/SemEval16_gold_Laptops/sample.txt"
    # TODO num_classes should be the number of unique categories in the dataset
    # TODO figure out how to dynamically give this input
    dataset_info = {"name": "SemEval", "num_classes": 5}
    trainset = GraphDataset(file_path=file_path, dataset_info=dataset_info)
    # testset = MiniGCDataset(80, 10, 20)

    # # Use PyTorch's DataLoader and the collate function defined before.
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                             collate_fn=batch_graphs)

    # Create model
    model = GAT_Graph_Classifier(in_feats, hid_feats, num_heads=num_heads,
                                 n_classes=trainset.num_classes)
    # logger.info(model)

    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_graph_classifier(model, data_loader, loss_func=loss_func,
                           optimizer=optimizer, epochs=5)


def main():

    graph_multiclass_classification(in_feats=1, hid_feats=4, num_heads=2)


if __name__ == "__main__":
    main()
