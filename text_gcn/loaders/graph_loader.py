import string
import json
import re
import pathlib

import numpy as np
import pandas as pd
import spacy
from scipy.sparse import lil_matrix
import torch.utils.data
from xml.etree import ElementTree as ET

from ..utils import TextProcessing
from ..utils import graph_utils
from ..utils import utils
from ..graph import DGL_Graph

from config import configuration as cfg
from logger.logger import logger

from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

nlp = spacy.load('en_core_web_sm')

# TODO clean sem_eval_14 dataset -> labels like windows_xp to operating system


def _clean_term(label):
    label = label.lower()
    all_stopwords = nlp.Defaults.stop_words
    label_tokens = nlp(label)
    label = " ".join([str(word) for word in label_tokens if word not in all_stopwords])
    label = re.sub('[^A-Za-z0-9 ]+', '', label)
    label = label[:30]
    return label


def _custom_one_hot_vector(labels_doc_dict_list):
    """
    Creates custom one hot vector for the labels
    Args:
        labels_doc_dict_list: A list of dictionary containing labels
    """
    label_text_to_label_id = {}
    label_counter = 0
    for labels in labels_doc_dict_list:
        for label in labels.keys():
            try:
                label_text_to_label_id[label]
            except KeyError:
                label_text_to_label_id[label] = label_counter
                label_counter += 1
    zero_vector = [-2 for i in range(len(label_text_to_label_id))]
    custom_one_hot_vector = [zero_vector[:] for i in range(len(labels_doc_dict_list))]

    for i, labels in enumerate(labels_doc_dict_list):
        for label in labels.keys():
            custom_one_hot_vector[i][label_text_to_label_id[label]] = labels[label]

    return custom_one_hot_vector, label_text_to_label_id


def _parse_sem_eval_14_type(dataset_path, text_processor=None, label_type="term"):
    """
    Parses sem eval 14 format dataset
    Args:
        file_name: file containing the sem eval dataset in XML format

    Returns:
        parsed_data: pandas dataframe for the parsed
        data containing labels and text
    """
    tree = ET.parse(dataset_path)
    root = tree.getroot()
    data_row = []
    labels_doc_dict_list = []
    for sentence in root.findall('sentence'):
        rid = sentence.get('id')
        temp_row = [rid, 'lorem ipsum']
        text = ''
        labels_dict = {}
        if label_type == "term":
            aspect_object = sentence.findall('aspectTerms')
        else:
            aspect_object = sentence.findall('aspectCategories')
        if len(aspect_object) == 0:
            continue

        found_text = sentence.find('text').text
        if cfg['DEBUG']:
            text += found_text + " "
        else:
            text += text_processor.process_text(found_text) + " "

        for aspects in aspect_object:
            for aspect in aspects:
                if label_type == "term":
                    label = aspect.get('term')
                    label = _clean_term(label)
                else:
                    label = aspect.get('category')
                polarity = aspect.get('polarity')
                polarity_int = 1 if polarity == "positive" else 0 if polarity == "neutral" else -1
                try:
                    if labels_dict[label] != polarity_int:
                        labels_dict[label] = 2
                except KeyError:
                    labels_dict[label] = polarity_int
        # include the review only if the number of words are more than 1
        if sum([i.strip(string.punctuation).isalpha() for i in text.split()]) > 1:
            labels_doc_dict_list += [labels_dict]
            temp_row[1] = text
            data_row += [temp_row]

    parsed_data = pd.DataFrame(data_row, columns=['id', 'text'])
    parsed_data['labels'], label_text_to_label_id = _custom_one_hot_vector(labels_doc_dict_list)
    return parsed_data, label_text_to_label_id


def _parse_sem_eval_16_type(dataset_path, text_processor=None):
    """
    Parses sem eval 16 format dataset
    Args:
        file_name: file containing the sem eval dataset in XML format

    Returns:
        parsed_data: pandas dataframe for the parsed
        data containing labels and text
    """
    tree = ET.parse(dataset_path)
    root = tree.getroot()
    data_row = []
    labels_doc_dict_list = []
    for review in root.findall('Review'):
        rid = review.get('rid')
        temp_row = [rid, 'lorem ipsum']
        text = ''
        labels_dict = {}
        for sentences in review:
            for sentence in sentences:
                opinions_object = sentence.findall('Opinions')
                if len(opinions_object) == 0:
                    continue

                found_text = sentence.find('text').text
                if cfg['DEBUG']:
                    text += found_text + " "
                else:
                    text += text_processor.process_text(found_text) + " "

                for opinions in opinions_object:
                    for opinion in opinions:
                        category = opinion.get('category').split('#')[0]
                        polarity = opinion.get('polarity')
                        polarity_int = 1 if polarity == "positive" else -1
                        try:
                            if labels_dict[category] != polarity_int:
                                labels_dict[category] = 2
                        except KeyError:
                            labels_dict[category] = polarity_int
        # include the review only if the number of words are more than 1
        if sum([i.strip(string.punctuation).isalpha() for i in text.split()]) > 1:
            labels_doc_dict_list += [labels_dict]
            temp_row[1] = text
            data_row += [temp_row]

    parsed_data = pd.DataFrame(data_row, columns=['id', 'text'])
    parsed_data['labels'], label_text_to_label_id = _custom_one_hot_vector(labels_doc_dict_list)
    return parsed_data, label_text_to_label_id


def _parse_twitter(dataset_path, text_processor):
    """
    Parses twitter dataset
    Args:
        file_name: file containing the twitter dataset

    Returns:
        parsed_data: pandas dataframe for the parsed data
        containing labels and text
    """
    count = 0
    data_row = []
    entity = "lorem ipsum"
    index = 0
    with open(dataset_path, "r") as file1:
        for line in file1:
            stripped_line = line.strip()
            if count % 3 == 0:
                temp_row = [index, 'lorem ipsum', []]
                temp_row[1] = text_processor.process_text(stripped_line)
                index += 1
            elif count % 3 == 1:
                entity = stripped_line
            elif count % 3 == 2:
                temp_row[1] += [int(stripped_line)]
                # replace $T$ with entity in temp_row[0]
                temp_row[0] = re.sub(r'$T$', entity, temp_row[0])
                data_row += [temp_row]
            count += 1
    parsed_data = pd.DataFrame(data_row, columns=['id', 'text', 'labels'])
    return parsed_data


class GraphDataset(torch.utils.data.Dataset):
    """
    Class for parsing data from files and storing dataframe
    """

    def __init__(self, graphs=None, labels=None, dataframe_df_path=None, label_text_to_label_id_path=None,
                 dataset_path=None, dataset_info=None, graph_path=None):
        """
        Initializes the loader
        Args:
            dataset_path: Path to the file containing the dataset.
            dataset_info: Dictionary consisting of one field 'dataset_name'
            graph_path: path to the bin file containing the saved DGL graph
        """

        assert ((dataframe_df_path is not None and label_text_to_label_id_path is not None)
                or (graph_path is not None and label_text_to_label_id_path is not None)
                or (dataset_path is not None and dataset_info is not None)
                or (graphs is not None and labels is not None)), \
            "Either labels and graphs array should be given or graph_path \
            should be specified or dataset_path and dataset_info should be specified"

        self.dataset_info = dataset_info
        if graphs is not None and labels is not None:
            self.graphs = graphs
            self.labels = labels
            self.classes = len(self.labels)

        elif (graph_path is None and dataframe_df_path is None) or cfg['training']['create_dataset']:
            assert pathlib.Path(dataset_path).exists(), "dataset path is not valid!"
            self.dataset_path = dataset_path

            if cfg['DEBUG']:
                self.text_processor = None
            else:
                self.text_processor = TextProcessing()

            df, label_text_to_label_id = self.get_dataset_df(self.text_processor)
            df, label_text_to_label_id = self.prune_dataset_df(df, label_text_to_label_id)
            self.save_label_text_to_label_id_dict(label_text_to_label_id)
            self.save_dataset_df(df)

            self.print_dataframe_statistics(df, label_text_to_label_id)

            token_graph_ob = DGL_Graph(df)
            self.graphs, labels_dict = token_graph_ob.create_instance_dgl_graphs()
            token_graph_ob.save_graphs(cfg['paths']['data_root'] + dataset_info['name'] + "_train_graph.bin",
                                       self.graphs, labels_dict)

            self.labels = labels_dict["glabel"]

        elif graph_path is not None:
            # If DGL graph is given in a file
            assert pathlib.Path(graph_path).exists(), "graph path is not valid!"
            assert pathlib.Path(label_text_to_label_id_path).exists(), "Label to id path is not valid!"

            label_text_to_label_id = self.read_label_text_to_label_id_dict(label_text_to_label_id_path)

            try:
                df = pd.read_csv(dataframe_df_path)
                logger.info("Read dataframe from " + dataframe_df_path)
                self.print_dataframe_statistics(df, label_text_to_label_id)
            except Exception:
                pass

            self.graphs, self.labels = graph_utils.load_dgl_graphs(graph_path)
            logger.info("Read graphs from " + graph_path)

        else:
            # If the dataframe is given
            assert pathlib.Path(dataframe_df_path).exists(), "dataframe path is not valid!"
            assert pathlib.Path(label_text_to_label_id_path).exists(), "Label to id path is not valid!"

            df = pd.read_csv(dataframe_df_path)
            logger.info("Read dataframe from " + dataframe_df_path)
            label_text_to_label_id = self.read_label_text_to_label_id_dict(label_text_to_label_id_path)
            self.print_dataframe_statistics(df, label_text_to_label_id)

            token_graph_ob = DGL_Graph(df)
            self.graphs, labels_dict = token_graph_ob.create_instance_dgl_graphs()
            token_graph_ob.save_graphs(cfg['paths']['data_root'] + dataset_info['name'] + "_train_graph.bin",
                                       self.graphs, labels_dict)

            self.labels = labels_dict["glabel"]
        # atleast one document is expected
        self.classes = len(self.labels[0])

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
        (dgl.DGLGraph, one hot vector)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]

    def get_graphs(self):
        """
        Returns the graphs array
        """
        return list(self.graphs)

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
            return _parse_twitter(self.dataset_path, text_processor)
        elif dataset_name == "SemEval16":
            return _parse_sem_eval_16_type(self.dataset_path, text_processor)
        elif dataset_name == "FourSquared":
            return _parse_sem_eval_16_type(self.dataset_path, text_processor)
        elif dataset_name == "SemEval14":
            return _parse_sem_eval_14_type(self.dataset_path, text_processor)
        elif dataset_name == "MAMS_ACSA":
            return _parse_sem_eval_14_type(self.dataset_path, text_processor, label_type="category")
        elif dataset_name == "MAMS_ATSA":
            return _parse_sem_eval_14_type(self.dataset_path, text_processor)
        elif dataset_name == "SamsungGalaxy":
            return _parse_sem_eval_14_type(self.dataset_path, text_processor)
        else:
            logger.error("{} dataset not yet supported".format(self.dataset_name))
            return NotImplemented

    def save_label_text_to_label_id_dict(self, label_text_to_label_id):
        with open(cfg['paths']['data_root'] + self.dataset_info['name'] + "_label_text_to_label_id.json", "w") as f:
            json_dict = json.dumps(label_text_to_label_id)
            f.write(json_dict)
        logger.info("Generated Label to ID mapping and stored at " + cfg['paths']['data_root'])

    def save_dataset_df(self, df):
        df.to_csv(cfg['paths']['data_root'] + self.dataset_info['name'] + "_dataframe.csv")
        logger.info("Generated dataframe and stored at " + cfg['paths']['data_root'])

    def read_label_text_to_label_id_dict(self, label_text_to_label_id_path):
        with open(label_text_to_label_id_path, "r") as f:
            label_text_to_label_id = json.load(f)
        logger.info("Read label to id mapping from " + label_text_to_label_id_path)
        return label_text_to_label_id

    def print_dataframe_statistics(self, df, label_text_to_label_id):

        utils.print_dataframe_statistics(df, label_text_to_label_id)

    def prune_dataset_df(self, df, label_text_to_label_id):

        df, label_text_to_label_id = utils.prune_dataset_df(df, label_text_to_label_id)
        return df, label_text_to_label_id

    def split_data(self, test_size=0.3, stratified=False, random_state=0, order=2):
        """
        Splits dataframe into train and test with optional stratified splitting
        returns 2 GraphDataset, one for train, one for test
        """

        graphs = self.get_graphs()
        sample_keys = lil_matrix(np.reshape(np.arange(len(graphs)), (len(graphs), -1)))

        labels = self.get_labels()
        labels = lil_matrix(np.array(labels))

        if stratified is False:
            x_split, y_split, x_labels, y_labels = train_test_split(sample_keys, labels, test_size=test_size, random_state=random_state)
        else:
            x_split, x_labels, y_split, y_labels = iterative_train_test_split(sample_keys, labels, test_size=test_size)

        x_labels = x_labels.todense().tolist()
        y_labels = y_labels.todense().tolist()
        x_split = x_split.todense().tolist()
        y_split = y_split.todense().tolist()

        x_graphs = list(map(lambda index: graphs[index[0]], x_split))
        y_graphs = list(map(lambda index: graphs[index[0]], y_split))

        x = GraphDataset(x_graphs, x_labels)
        y = GraphDataset(y_graphs, y_labels)

        return x, y
