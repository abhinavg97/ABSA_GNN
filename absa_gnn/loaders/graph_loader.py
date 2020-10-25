import json
import pathlib

import numpy as np
import pandas as pd
import torch.utils.data
from scipy.sparse import lil_matrix

from ast import literal_eval

from ..utils import TextProcessing
from ..utils import graph_utils
from ..utils import utils
from ..utils import parsing
from ..graph import DGL_Graph

from config import configuration as cfg
from logger.logger import logger


class GraphDataset(torch.utils.data.Dataset):
    """
    Class for parsing data from files and storing dataframe
    """

    def __init__(self, graphs=None, labels=None, dataset_df_path=None, label_text_to_label_id_path=None,
                 dataset_path=None, dataset_info=None, train_graphs_path=None, large_graph_path=None):
        """
        Initializes the loader
        Args:
            graphs:                      list of DGL graphs
            labels:                      list of labels corresponding to the above DGL graphs
            dataset_df_path:             dataframe path for the processed dataset
            label_text_to_label_id_path: path of json file containing label to id mapping for the dataset
            dataset_path:                path to the file containing the dataset.
            dataset_info:                dictionary consisting of one field 'dataset_name'
            train_graphs_path:           path to the bin file containing the saved DGL graph
            large_graph_path:            path of the large graph for the dataset
        """

        assert ((graphs is not None and labels is not None)
                or (pathlib.Path(train_graphs_path).is_file() and pathlib.Path(large_graph_path).is_file()
                    and pathlib.Path(label_text_to_label_id_path).is_file())
                or (pathlib.Path(dataset_df_path).is_file() and pathlib.Path(label_text_to_label_id_path).is_file())
                or (pathlib.Path(dataset_path).is_file() and dataset_info is not None)
                ), \
            "Either labels and graphs array should be given or\
            train_graphs_path and large_graph_path should be specified or \
            dataframe_path should be specified\
            or dataset_path and dataset_info should be specified "

        self.dataset_info = dataset_info

        # If the graphs and labels are provided in an array format
        if graphs is not None and labels is not None:
            self.graphs = graphs
            self.labels = labels
            self.classes = len(self.labels)

        # If the graphs stored in a file is given
        elif pathlib.Path(train_graphs_path).is_file() and pathlib.Path(large_graph_path).is_file() \
                and not cfg['training']['create_dataset']:

            assert pathlib.Path(label_text_to_label_id_path).exists(), "Label text to label id path is not valid!"

            label_text_to_label_id = self.read_label_text_to_label_id_dict(label_text_to_label_id_path)

            try:
                df = pd.read_csv(dataset_df_path, index_col=0)
                df['labels'] = list(map(lambda label_list: literal_eval(label_list), df['labels'].tolist()))
                logger.info("Reading dataframe from " + dataset_df_path)
                self.print_dataframe_statistics(df, label_text_to_label_id)
            except Exception:
                pass

            self.graphs, self.labels = graph_utils.load_dgl_graphs(train_graphs_path)
            logger.info("Reading train graphs from " + train_graphs_path)

            self.local_large_graph, __ = graph_utils.load_dgl_graphs(large_graph_path)
            logger.info("Reading large graph from " + large_graph_path)

        # If the dataset dataframe is given
        elif pathlib.Path(dataset_df_path).is_file() and not cfg['training']['create_dataset']:
            # If the dataframe is given
            assert pathlib.Path(label_text_to_label_id_path).is_file(), "Label text to label id path is not valid!"

            logger.info("Reading dataframe from " + dataset_df_path)
            df = pd.read_csv(dataset_df_path, index_col=0)
            df['labels'] = list(map(lambda label_list: literal_eval(label_list), df['labels'].tolist()))

            label_text_to_label_id = self.read_label_text_to_label_id_dict(label_text_to_label_id_path)

            self._finalize_graph_processing(dataset_info, df, label_text_to_label_id)

        # Otherwise make the graphs from raw data
        else:
            assert pathlib.Path(dataset_path).is_file(), "dataset path is not valid!"
            self.dataset_path = dataset_path

            if cfg['DEBUG']:
                self.text_processor = None
            else:
                self.text_processor = TextProcessing()

            df, label_text_to_label_id = self.get_dataset_df(self.text_processor)
            df, label_text_to_label_id = self.prune_dataset_df(df, label_text_to_label_id)
            self.save_label_text_to_label_id_dict(label_text_to_label_id)
            self.save_dataset_df(df)

            self._finalize_graph_processing(dataset_info, df, label_text_to_label_id)

        # atleast one document is expected
        self.classes = len(self.labels[0])

    def _finalize_graph_processing(self, dataset_info,  df, label_text_to_label_id):

        self.print_dataframe_statistics(df, label_text_to_label_id)
        token_graph_ob = DGL_Graph(df)
        self.graphs, labels_dict = token_graph_ob.create_instance_dgl_graphs()
        self.local_large_graph = token_graph_ob.create_large_dgl_graph()

        token_graph_ob.save_graphs(cfg['paths']['data_root'] + dataset_info['name'] + '_train_graphs.bin', self.graphs, labels_dict)

        token_graph_ob.save_graphs(cfg['paths']['data_root'] + dataset_info['name'] + '_large_graph.bin', self.local_large_graph)

        self.labels = labels_dict["glabel"]

    @property
    def num_classes(self):
        """Number of classes."""
        return self.classes

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, multi hot vector)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]

    def get_graphs(self):
        """
        Returns the graphs array
        """
        return list(self.graphs)

    @property
    def large_graph(self):
        """
        Returns the large graph
        """
        return self.local_large_graph

    def get_labels(self):
        """
        Return the labels
        """
        return list(self.labels)

    def get_dataset_df(self, text_processor=None):
        """
        Returns pandas dataframe, label to id mapping
        INDEX           TEXT                   LABELS
          0       "Example sentence"     [-1, -2, 2, 0, 1]

         -2 -> label not present in the text
         -1 -> negative sentiment
          0 -> neutral sentiment
          1 -> positive sentiment
          2 -> ambiguous sentiment

          Label to ID mapping maps index of label list to the label name
        """
        if text_processor is None:
            text_processor = self.text_processor
        dataset_name = self.dataset_info["name"]
        if dataset_name == "Twitter":
            return parsing._parse_twitter(self.dataset_path, text_processor)
        elif dataset_name == "SemEval16":
            return parsing._parse_sem_eval_16_type(self.dataset_path, text_processor)
        elif dataset_name == "FourSquared":
            return parsing._parse_sem_eval_16_type(self.dataset_path, text_processor)
        elif dataset_name == "SemEval14":
            return parsing._parse_sem_eval_14_type(self.dataset_path, text_processor)
        elif dataset_name == "MAMS_ACSA":
            return parsing._parse_sem_eval_14_type(self.dataset_path, text_processor, label_type="category")
        elif dataset_name == "MAMS_ATSA":
            return parsing._parse_sem_eval_14_type(self.dataset_path, text_processor)
        elif dataset_name == "SamsungGalaxy":
            return parsing._parse_sem_eval_14_type(self.dataset_path, text_processor)
        elif dataset_name == "20NG":
            return parsing._parse_text_gcn_type(self.dataset_path)
        elif dataset_name == "MR":
            return parsing._parse_text_gcn_type(self.dataset_path)
        elif dataset_name == "Ohsumed":
            return parsing._parse_text_gcn_type(self.dataset_path)
        elif dataset_name == "R8":
            return parsing._parse_text_gcn_type(self.dataset_path)
        elif dataset_name == "R52":
            return parsing._parse_text_gcn_type(self.dataset_path)
        else:
            logger.error("{} dataset not yet supported".format(self.dataset_name))
            return NotImplemented

    def save_label_text_to_label_id_dict(self, label_text_to_label_id):
        with open(cfg['paths']['data_root'] + self.dataset_info['name'] + "_label_text_to_label_id.json", "w") as f:
            json_dict = json.dumps(label_text_to_label_id)
            f.write(json_dict)
        logger.info("Generated Label to ID mapping and stored at " + cfg['paths']['data_root'])

    def save_dataset_df(self, df):
        df.drop(columns=['id'], inplace=True)
        df.to_csv(cfg['paths']['data_root'] + self.dataset_info['name'] + "_dataframe.csv", index_label='id')
        logger.info("Generated dataframe and stored at " + cfg['paths']['data_root'])

    def read_label_text_to_label_id_dict(self, label_text_to_label_id_path):
        with open(label_text_to_label_id_path, "r") as f:
            label_text_to_label_id = json.load(f)
        logger.info("Reading label to id mapping from " + label_text_to_label_id_path)
        return label_text_to_label_id

    def print_dataframe_statistics(self, df, label_text_to_label_id):

        utils.print_dataframe_statistics(df, label_text_to_label_id)

    def prune_dataset_df(self, df, label_text_to_label_id):

        df, label_text_to_label_id = utils.prune_dataset_df(df, label_text_to_label_id)
        return df, label_text_to_label_id

    def split_data(self, test_size=0.3, stratified=False, random_state=0):
        """
        Splits dataframe into train and test with optional stratified splitting
        returns 2 GraphDataset, one for train, one for test
        """

        graphs = self.get_graphs()
        sample_keys = lil_matrix(np.reshape(np.arange(len(graphs)), (len(graphs), -1)))

        labels = self.get_labels()
        labels_for_split = list(map(lambda label_vec: list(map(lambda x: 0 if x == -2 else 1, label_vec)), labels))

        labels_for_split = lil_matrix(np.array(labels_for_split))

        _, _, x_split, y_split = utils.split_data(sample_keys=sample_keys, labels=labels_for_split,
                                                  test_size=test_size, stratified=stratified,
                                                  random_state=random_state)

        x_graphs = list(map(lambda index: graphs[index[0]], x_split))
        y_graphs = list(map(lambda index: graphs[index[0]], y_split))

        x_labels = list(map(lambda index: labels[index[0]], x_split))
        y_labels = list(map(lambda index: labels[index[0]], y_split))

        x = GraphDataset(x_graphs, x_labels)
        y = GraphDataset(y_graphs, y_labels)

        return x, y
