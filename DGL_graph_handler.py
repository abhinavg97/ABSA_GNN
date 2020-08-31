import torch
from dgl import batch as g_batch
import torch.optim as optim
from torch.utils.data import DataLoader
from text_gcn.loaders import GraphDataset
from text_gcn.models import GAT_Graph_Classifier
from text_gcn.trainers import Trainer


def batch_graphs(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = g_batch(graphs)
    return batched_graph, torch.tensor(labels)


in_feats = 1
hid_feats = 4
num_heads = 2

# Create training and test sets.
graph_path = "./output/graph.bin"
file_path = "/home/abhi/Desktop/gcn/data/SemEval16_gold_Laptops/sample.txt"

# TODO take the dataset_info info from config file

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataloader initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Dataset is decoupled to allow more flexibility in choosing different types of dataset to train
dataset_info = {"name": "SemEval"}
trainset = GraphDataset(graph_path=graph_path, dataset_info=dataset_info)
# testset = MiniGCDataset(80, 10, 20)

# Use PyTorch's DataLoader and the collate function defined before.
data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=batch_graphs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create the layer
model = GAT_Graph_Classifier(in_feats, hid_feats, num_heads=num_heads,
                             n_classes=trainset.num_classes)
# logger.info(model)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Trainer Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

loss_func = torch.nn.BCEWithLogitsLoss()
trainer = Trainer(model, loss_func, data_loader)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

epochs = 5
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer.train(epochs, optimizer)
