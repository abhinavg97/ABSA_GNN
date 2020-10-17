import pathlib
import json

import torch
import torch.optim as optim
import torch.nn.functional as F

from dgl import mean_nodes
from dgl.nn.pytorch.conv import GATConv

import pytorch_lightning as pl

from ..metrics import class_wise_f1_scores, class_wise_precision_scores, class_wise_recall_scores,\
                      f1_score, precision_score, recall_score, accuracy_score

from config import configuration as cfg


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

    def configure_optimizers(self, lr=cfg['training']['optimizer']['learning_rate']):
        return optim.Adam(self.parameters(), lr=lr)

    def loss_function(self, prediction, label):
        return F.binary_cross_entropy_with_logits(prediction, label)

    def shared_step(self, batch):
        """
        shared step between train, val and test to calculate the loss
        """
        graph_batch, labels = batch
        # convert labels to 1's if label value is present else convert to 0
        # This is to predict the aspect given text
        labels = torch.Tensor(list(map(lambda label_vec: list(map(lambda x: 0 if x == -2 else 1, label_vec)), labels)))
        prediction = self(graph_batch)
        labels = labels.type_as(prediction)
        loss = self.loss_function(prediction, labels)
        return loss, prediction

    def _calc_metrics(self, outputs):
        """
        helper function to calculate the metrics
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        predictions = torch.Tensor()
        labels = torch.LongTensor()
        for x in outputs:
            predictions = torch.cat((predictions, x['prediction']), 0)
            labels = torch.cat((labels, x['labels']), 0)

        labels = torch.Tensor(list(map(lambda label_vec: list(map(lambda x: 0 if x == -2 else 1, label_vec)), labels)))

        avg_f1_score = f1_score(labels, predictions)
        avg_recall_score = recall_score(labels, predictions)
        avg_precision_score = precision_score(labels, predictions)
        avg_accuracy_score = accuracy_score(labels, predictions)

        class_f1_scores_list = class_wise_f1_scores(labels, predictions)
        class_precision_scores_list = class_wise_precision_scores(labels, predictions)
        class_recall_scores_list = class_wise_recall_scores(labels, predictions)

        class_f1_scores = {}
        class_precision_scores = {}
        class_recall_scores = {}

        try:
            label_text_to_label_id_path = cfg['paths']['data_root'] + cfg['paths']['label_text_to_label_id']
            assert pathlib.Path(label_text_to_label_id_path).exists(), "Label to id path is not valid! Using Incremental class names"
            with open(label_text_to_label_id_path, "r") as f:
                label_text_to_label_id = json.load(f)
            label_id_to_label_text = {value: key for key, value in label_text_to_label_id.items()}

        except Exception:
            label_id_to_label_text = {i: f'class_{i}' for i in range(len(class_f1_scores_list))}

        for index in range(len(label_id_to_label_text)):
            class_f1_score = class_f1_scores_list[index]
            class_precision_score = class_precision_scores_list[index]
            class_recall_score = class_recall_scores_list[index]

            class_name = label_id_to_label_text[index]

            class_f1_scores[class_name] = class_f1_score
            class_precision_scores[class_name] = class_precision_score
            class_recall_scores[class_name] = class_recall_score

        return avg_loss, class_f1_scores, class_precision_scores, class_recall_scores,\
            avg_f1_score, avg_precision_score, avg_recall_score, avg_accuracy_score

    def training_step(self, batch, batch_idx):
        graph_batch, labels = batch
        batch_train_loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        return {'loss': batch_train_loss, 'prediction': prediction, 'labels': labels}

    def training_epoch_end(self, outputs):
        avg_train_loss, class_f1_scores, class_precision_scores, class_recall_scores,\
         avg_f1_score, avg_precision_score, avg_recall_score, avg_accuracy_score = self._calc_metrics(outputs)

        self.logger.log_metrics(metrics={'avg_train_loss': avg_train_loss}, step=self.trainer.current_epoch)
        self.logger.log_metrics(metrics={'avg_train_f1_score': avg_f1_score}, step=self.trainer.current_epoch)
        self.logger.log_metrics(metrics={'avg_train_precision_score': avg_precision_score}, step=self.trainer.current_epoch)
        self.logger.log_metrics(metrics={'avg_train_recall_score': avg_recall_score}, step=self.trainer.current_epoch)
        self.logger.log_metrics(metrics={'avg_train_accuracy_score': avg_accuracy_score}, step=self.trainer.current_epoch)
        self.logger.experiment.add_scalars('train_class_f1_scores', class_f1_scores, global_step=self.trainer.current_epoch)
        self.logger.experiment.add_scalars('train_class_precision_scores', class_precision_scores, global_step=self.trainer.current_epoch)
        self.logger.experiment.add_scalars('train_class_recall_scores', class_recall_scores, global_step=self.trainer.current_epoch)

    def validation_step(self, batch, batch_idx):
        graph_batch, labels = batch
        batch_val_loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        return {'loss': batch_val_loss, 'prediction': prediction, 'labels': labels}

    def validation_epoch_end(self, outputs):
        avg_val_loss, class_f1_scores, class_precision_scores, class_recall_scores,\
         avg_f1_score, avg_precision_score, avg_recall_score, avg_accuracy_score = self._calc_metrics(outputs)

        # val_f1_score is logged via self.log also so that PL can use early stopping
        self.log('val_f1_score', avg_f1_score)
        self.logger.log_metrics(metrics={'avg_val_loss': avg_val_loss}, step=self.trainer.current_epoch)
        self.logger.log_metrics(metrics={'avg_val_f1_score': avg_f1_score}, step=self.trainer.current_epoch)
        self.logger.log_metrics(metrics={'avg_val_precision_score': avg_precision_score}, step=self.trainer.current_epoch)
        self.logger.log_metrics(metrics={'avg_val_recall_score': avg_recall_score}, step=self.trainer.current_epoch)
        self.logger.log_metrics(metrics={'avg_val_accuracy_score': avg_accuracy_score}, step=self.trainer.current_epoch)
        self.logger.experiment.add_scalars('val_class_f1_scores', class_f1_scores, global_step=self.trainer.current_epoch)
        self.logger.experiment.add_scalars('val_class_precision_scores', class_precision_scores, global_step=self.trainer.current_epoch)
        self.logger.experiment.add_scalars('val_class_recall_scores', class_recall_scores, global_step=self.trainer.current_epoch)

    def test_step(self, batch, batch_idx):
        graph_batch, labels = batch
        batch_test_loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        return {'loss': batch_test_loss, 'prediction': prediction, 'labels': labels}

    def test_epoch_end(self, outputs):
        avg_test_loss, class_f1_scores, class_precision_scores, class_recall_scores,\
         avg_f1_score, avg_precision_score, avg_recall_score, avg_accuracy_score = self._calc_metrics(outputs)

        self.logger.log_metrics(metrics={'avg_test_loss': avg_test_loss}, step=0)
        self.logger.log_metrics(metrics={'avg_test_f1_score': avg_f1_score}, step=0)
        self.logger.log_metrics(metrics={'avg_test_precision_score': avg_precision_score}, step=0)
        self.logger.log_metrics(metrics={'avg_test_recall_score': avg_recall_score}, step=0)
        self.logger.log_metrics(metrics={'avg_test_accuracy_score': avg_accuracy_score}, step=0)
        self.logger.experiment.add_scalars('test_class_f1_scores', class_f1_scores, global_step=0)
        self.logger.experiment.add_scalars('test_class_precision_scores', class_precision_scores, global_step=0)
        self.logger.experiment.add_scalars('test_class_recall_scores', class_recall_scores, global_step=0)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

# TODO check large update dgl graph from layers file
