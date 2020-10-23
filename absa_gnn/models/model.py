import pathlib
import json

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


from ..layers import Instance_Graphs_GAT, MatrixUpdation
from ..metrics import class_wise_f1_scores, class_wise_precision_scores, class_wise_recall_scores,\
                      f1_score, precision_score, recall_score, accuracy_score


from config import configuration as cfg


class Model(pl.LightningModule):
    """
    GAT model class: This is where the learning happens
    The boilerplate for learning is abstracted away by Lightning
    """
    def __init__(self, in_dim, hidden_dim, num_heads, out_dim, num_classes, large_graph):
        super(Model, self).__init__()

        self.large_graph = large_graph
        self.large_token_gcn = MatrixUpdation(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        self.small_instance_gat = Instance_Graphs_GAT(in_dim=in_dim, hidden_dim=hidden_dim, num_heads=num_heads, out_dim=out_dim)
        # We may have different out_dim's from 2 GNNs and concat them.

        self.classify = torch.nn.Linear(2 * out_dim, num_classes)

    def forward(self, small_batch_graphs, combine='concat'):
        """ Combines embeddings of tokens from large and small graph by concatenating.

        Take embeddings from large graph for tokens present in the small graph batch.
        Need token to index information to fetch from large graph.
        Arrange small tokens and large tokens in same order.

        small_batch_graphs: Instance graph batch
        small_batch_embs: Embeddings from instance GAT
        token_idx_batch: Ordered set of token ids present in the current batch of instance graph
        large_graph: Large token graph
        large_embs: Embeddings from large GCN
        combine: How to combine two embeddings (Default: concatenate)

        Should be converted to set of unique tokens before fetching from large graph.
        Convert to boolean mask indicating the tokens present in the current batch of instances.
        Boolean mask size: List of number of tokens in the large graph.
        """
        # Fetch embeddings from instance graphs:
        token_embs_small = self.small_instance_gat(small_batch_graphs, small_batch_graphs.ndata['emb'])

        # Fetch embeddings from large token graph
        token_embs_large = self.large_token_gcn(self.large_graph, self.large_graph.ndata['emb'])

        token_idx_batch = small_batch_graphs.ndata['item_id']

        # Fetch embeddings from large graph of tokens present in instance batch only:
        token_embs_large = token_embs_large[token_idx_batch]

        # Combines both embeddings:
        if combine == 'concat':
            embs = torch.cat([token_embs_small, token_embs_large])
        elif combine == 'avg':
            embs = torch.mean(torch.stack([token_embs_small, token_embs_large]), dim=0)
        else:
            raise NotImplementedError(f'combine supports either concat or avg.'
                                      f' [{combine}] provided.')
        return self.classify(embs)

    def configure_optimizers(self, lr=cfg['training']['optimizer']['learning_rate']):
        return torch.optim.Adam(self.parameters(), lr=lr)

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


# TODO set a flag in large graph to create doc to doc edge
# TODO prepare the text gcn datasets
