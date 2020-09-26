import pytorch_lightning as pl
from ..loaders import GraphDataset
from torch.utils.data import DataLoader
from dgl import batch as g_batch
import torch
from config import configuration as cfg


# When doing distributed training, Datamodules have two optional arguments for
# granular control over download/prepare/splitting data:


class GraphDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning datamodule to handle training, testing and validation dataloaders
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """

    def __init__(self, train_batch_size=cfg['training']['train_batch_size'],
                 val_batch_size=cfg['training']['val_batch_size'],
                 test_batch_size=cfg['training']['test_batch_size'],
                 data_root=cfg['paths']['data_root'], dataset=cfg['paths']['dataset'],
                 dataframe=cfg['paths']['dataframe'], label_text_to_label_id=cfg['paths']['label_text_to_label_id'],
                 graph=cfg['paths']['saved_graph'], dataset_info=cfg['data']['dataset']):
        super().__init__()
        self.dataset_info = dataset_info
        self.graph_data = GraphDataset(
            dataframe_df_path=data_root + dataframe,
            label_text_to_label_id_path=data_root
                                        + label_text_to_label_id,
            dataset_path=data_root + dataset, graph_path=data_root + graph,
            dataset_info=self.dataset_info)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def batch_graphs(self, samples):
        """
        The input `samples` is a list of pairs (graph, label).
        """
        graphs, labels = map(list, zip(*samples))
        batched_graph = g_batch(graphs)
        return batched_graph, torch.tensor(labels)

    # def prepare_data(self):
    #     MNIST(os.getcwd(), train=True, download=True)
    #     MNIST(os.getcwd(), train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        """
        Split data into train, val and test
        """
        trainval_test_split = cfg['data']['trainval_test_split']
        graph_train_val, self.graph_test = self.graph_data.split_data(
            test_size=trainval_test_split)

        train_val_split = cfg['data']['train_val_split']
        self.graph_train, self.graph_val = graph_train_val.split_data(
            test_size=train_val_split)

    def train_dataloader(self):
        """
        Return the dataloader for each split
        """
        # Use PyTorch's DataLoader and the collate function defined before.
        graph_train = DataLoader(self.graph_train, batch_size=self.train_batch_size, shuffle=True,
                                 collate_fn=self.batch_graphs)
        return graph_train

    def val_dataloader(self):
        graph_val = DataLoader(self.graph_val, batch_size=self.val_batch_size, collate_fn=self.batch_graphs)
        return graph_val

    def test_dataloader(self):
        graph_test = DataLoader(self.graph_test, batch_size=self.test_batch_size, collate_fn=self.batch_graphs)
        return graph_test

    @property
    def num_classes(self):
        """
        Num classes is needed to initiate the model
        Therefore we return num_classes from the graph_data we defined in init
        """
        return self.graph_data.num_classes
