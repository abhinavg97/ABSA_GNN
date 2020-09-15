from xml.etree import ElementTree as ET
import pandas as pd
from logger.logger import logger
from ..utils import TextProcessing
from ..utils import graph_utils
from ..utils import utils
import re
from ..graph import DGL_Graph
import torch.utils.data
from config import configuration as cfg
import json
import pathlib
import string


def _custom_one_hot_vector(labels_doc_dict_list):
    """
    Creates custom one hot vector for the labels
    Args:
        labels_doc_dict_list: A list of dictionary containing labels
    """
    label_to_id = {}
    label_counter = 0
    for labels in labels_doc_dict_list:
        for label in labels.keys():
            try:
                label_to_id[label]
            except KeyError:
                label_to_id[label] = label_counter
                label_counter += 1
    zero_vector = [-2 for i in range(len(label_to_id))]
    custom_one_hot_vector = [zero_vector[:] for i in range(len(labels_doc_dict_list))]

    for i, labels in enumerate(labels_doc_dict_list):
        for label in labels.keys():
            custom_one_hot_vector[i][label_to_id[label]] = labels[label]

    return custom_one_hot_vector, label_to_id

# TODO clean sem_eval_14 dataset -> labels like windows_xp to operating system


def _parse_sem_eval_14(dataset_path, text_processor=None):
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
        aspect_terms_object = sentence.findall('aspectTerms')
        if len(aspect_terms_object) == 0:
            continue
        text += sentence.find('text').text + " "
        for aspect_terms in aspect_terms_object:
            for aspect_term in aspect_terms:
                term = aspect_term.get('term')
                polarity = aspect_term.get('polarity')
                polarity_int = 1 if polarity == "positive" else 0 if polarity == "neutral" else -1
                try:
                    if labels_dict[term] != polarity_int:
                        labels_dict[term] = 2
                except KeyError:
                    labels_dict[term] = polarity_int
        # include the review only if the number of words are more than 1
        if sum([i.strip(string.punctuation).isalpha() for i in text.split()]) > 1:
            labels_doc_dict_list += [labels_dict]
            if cfg['DEBUG']:
                temp_row[1] = text
            else:
                temp_row[1] = text_processor.process_text(text)
            data_row += [temp_row]

    parsed_data = pd.DataFrame(data_row, columns=['id', 'text'])
    parsed_data['labels'], label_to_id = _custom_one_hot_vector(labels_doc_dict_list)
    return parsed_data, label_to_id


def _parse_sem_eval_16(dataset_path, text_processor=None):
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
                text += sentence.find('text').text + " "
                for opinions in opinions_object:
                    for opinion in opinions:
                        category = opinion.get('category')
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
            if cfg['DEBUG']:
                temp_row[1] = text
            else:
                temp_row[1] = text_processor.process_text(text)
            data_row += [temp_row]

    parsed_data = pd.DataFrame(data_row, columns=['id', 'text'])
    parsed_data['labels'], label_to_id = _custom_one_hot_vector(labels_doc_dict_list)
    return parsed_data, label_to_id


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

    def __init__(self, dataset_path=None, dataset_info=None, graph_path=None):
        """
        Initializes the loader
        Args:
            dataset_path: Path to the file containing the dataset.
            dataset_info: Dictionary consisting of one field 'dataset_name'
            graph_path: path to the bin file containing the saved DGL graph
        """
        assert (graph_path is not None or (dataset_path is not None and dataset_info is not None)), \
            "Either graph_path should be specified or dataset_path and dataset_info should be specified"
        self.dataset_info = dataset_info
        if graph_path is None or cfg['training']['create_dataset']:
            assert pathlib.Path(dataset_path).exists(), "dataset path is not valid!"
            self.dataset_path = dataset_path
            if not cfg['DEBUG']:
                text_processor = TextProcessing()
            else:
                text_processor = None
            df, self.label_to_id = self.get_dataset_df(text_processor)
            self.print_dataframe_statistics(df)
            token_graph_ob = DGL_Graph(df)
            self.graphs, labels_dict = token_graph_ob.create_instance_dgl_graphs()

            with open(cfg['paths']['data_root'] + dataset_info['name'] + "_label_to_id.json", "w") as f:
                json_dict = json.dumps(self.label_to_id)
                f.write(json_dict)

            token_graph_ob.save_graphs(cfg['paths']['data_root'] + dataset_info['name'] +
                                       "_train_graph.bin", self.graphs, labels_dict)
            self.labels = labels_dict["glabel"]
            logger.info("Graph and label_to_id generated and stored at " + cfg['paths']['data_root'])
        else:
            label_to_id_path = cfg['paths']['data_root'] + dataset_info['name'] + "_label_to_id.json"
            assert pathlib.Path(graph_path).exists(), "graph path is not valid!"
            assert pathlib.Path(label_to_id_path).exists(), "label to id information not available!"
            self.graphs, self.labels = graph_utils.load_dgl_graphs(graph_path)

            with open(label_to_id_path, "r") as f:
                self.label_to_id = json.load(f)
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
        return self.graphs

    def get_labels(self):
        """
        Return the labels
        """
        return self.labels

    def get_dataset_df(self, text_processor=None):
        """
        Returns pandas dataframe
        """
        dataset_name = self.dataset_info["name"]
        if(dataset_name == "Twitter"):
            return _parse_twitter(self.dataset_path, text_processor)
        elif(dataset_name == "SemEval16"):
            return _parse_sem_eval_16(self.dataset_path, text_processor)
        elif(dataset_name == "SemEval14"):
            return _parse_sem_eval_14(self.dataset_path, text_processor)
        else:
            logger.error(
                "{} dataset not yet supported".format(self.dataset_name))

    def print_dataframe_statistics(self, df):

        utils.print_dataframe_statistics(df, self.label_to_id)
