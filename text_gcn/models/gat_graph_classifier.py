import torch
from dgl.nn.pytorch.conv import GATConv
import torch.optim as optim
import torch.nn.functional as F
from dgl import mean_nodes
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import F1
from logger.logger import logger
from config import configuration as cfg


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

    def shared_step(self, batch):
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
        return loss, prediction

    def training_step(self, batch, batch_idx):
        loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        grap_batch, labels = batch
        val_loss, prediction = self.shared_step(batch)
        metric = F1()
        prediction = torch.sigmoid(prediction)
        f1_score = metric(prediction, labels)
        # TODO look at graphs of f1_score
        # result = pl.TrainResult(val_loss)
        # result.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        logger.info(prediction)
        return {'val_loss': val_loss, 'f1_score': f1_score}

    def validation_epoch_end(self, outputs):

        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        f1_score = torch.stack([x['f1_score'] for x in outputs]).mean()
        # self.logger.experiment._
        log = {'avg_val_loss': val_loss, 'f1_score_mean': f1_score}
        return {'log': log}

    def configure_optimizers(self, lr=cfg['training']['optimizer']['learning_rate']):
        return optim.Adam(self.parameters(), lr=lr)

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        return result
        # return prediction values after sigmoid
