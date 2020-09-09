import torch
from dgl.nn.pytorch.conv import GATConv
import torch.optim as optim
import torch.nn.functional as F
from dgl import mean_nodes
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from text_gcn.loaders import GraphDataset
from dgl import batch as g_batch
from pytorch_lightning.metrics.classification import F1


class GAT_Graph_Classifier(pl.LightningModule):
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

    def loss_function(self, prediction, label):
        return F.binary_cross_entropy_with_logits(prediction, label)

    def training_step(self, batch, batch_idx):
        graph_batch, labels = batch
        # convert labels to 1's if label value is not zero
        # This is to predict the aspect given text
        for label in labels:
            for i in range(len(label)):
                if label[i] == -1 or label[i] == 2:
                    label[i] = 1
        prediction = self(graph_batch)
        labels = labels.type_as(prediction)
        loss = self.loss_function(prediction, labels)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        graph_batch, labels = batch
        # convert labels to 1's if label value is not zero
        # This is to predict the aspect given text
        for label in labels:
            for i in range(len(label)):
                if label[i] == -1 or label[i] == 2:
                    label[i] = 1
        prediction = self(graph_batch)
        labels = labels.type_as(prediction)
        val_loss = self.loss_function(prediction, labels)
        metric = F1()
        f1_score = metric(prediction, labels)
        return {'val_loss': val_loss, 'f1_score': f1_score}

    def validation_epoch_end(self, outputs):

        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        f1_score = torch.stack([x['f1_score'] for x in outputs]).mean()
        # self.logger.experiment._
        log = {'avg_val_loss': val_loss, 'f1_score_mean': f1_score}
        return {'log': log}

    def configure_optimizers(self, lr=0.00001):
        # TODO lr as parameter to configure optimizers
        return optim.Adam(self.parameters(), lr=lr)

    def batch_graphs(self, samples):
        # The input `samples` is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = g_batch(graphs)
        return batched_graph, torch.tensor(labels)

    def train_dataloader(self):
        # Dataset is decoupled to allow more flexibility in choosing different types of dataset to train
        # TODO take dataset_info, graph_path and label_path from config file
        file_path = "/home/abhi/Desktop/gcn/data/SemEval16_gold_Laptops/sample.txt"
        graph_path = "./output/graph.bin"
        dataset_info = {"name": "SemEval"}
        trainset = GraphDataset(graph_path=graph_path, dataset_info=dataset_info)
        # Use PyTorch's DataLoader and the collate function defined before.
        train_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=self.batch_graphs)
        return train_loader

    def val_dataloader(self):
        # TODO take dataset_info, graph_path and label_path from config file
        file_path = "/home/abhi/Desktop/gcn/data/SemEval16_gold_Laptops/sample.txt"
        graph_path = "./output/graph.bin"
        dataset_info = {"name": "SemEval"}
        trainset = GraphDataset(graph_path=graph_path, dataset_info=dataset_info)
        # Use PyTorch's DataLoader and the collate function defined before.
        val_loader = DataLoader(trainset, batch_size=32, collate_fn=self.batch_graphs)
        return val_loader
