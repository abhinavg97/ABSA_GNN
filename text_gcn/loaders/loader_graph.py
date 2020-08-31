from xml.etree import ElementTree as ET
import pandas as pd
import logging
from ..utils import TextProcessing
from ..utils import graph_utils
import re
from ..graph import DGLGraph
import torch.utils.data


def _custom_one_hot_vector(labels_doc_dict_list):
    """
    Creates custom one hot vector for the labels
    Args:
        labels_doc_dict_list: A list of dictionary containing labels
    """
    # TODO save label_to_id in a file
    label_to_id = {}
    label_counter = 0
    for labels in labels_doc_dict_list:
        for label in labels.keys():
            try:
                label_to_id[label]
            except KeyError:
                label_to_id[label] = label_counter
                label_counter += 1
    zero_vector = [0 for i in range(len(label_to_id))]
    custom_one_hot_vector = [zero_vector for i in range(len(labels_doc_dict_list))]

    for i in range(len(labels_doc_dict_list)):
        labels = labels_doc_dict_list[i]
        for label in labels.keys():
            custom_one_hot_vector[i][label_to_id[label]] = labels[label]

    return custom_one_hot_vector


def _parse_sem_eval(file_path, text_processor):
    """
    Parses sem eval dataset
    Args:
        file_name: file containing the sem eval dataset in XML format

    Returns:
        parsed_data: pandas dataframe for the parsed
        data containing labels and text
    """
    tree = ET.parse(file_path)
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
                text += sentence.find('text').text + " "
                for opinions in sentence.findall('Opinions'):
                    for opinion in opinions:
                        category = opinion.get('category')
                        polarity = opinion.get('polarity')
                        polarity_int = 1 if polarity == "positive" else -1
                        try:
                            if labels_dict[category] != polarity_int:
                                labels_dict[category] = 2
                        except KeyError:
                            labels_dict[category] = polarity_int
        labels_doc_dict_list += [labels_dict]
        temp_row[1] = text_processor.process_text(text)
        data_row += [temp_row]

    parsed_data = pd.DataFrame(data_row, columns=['id', 'text'])
    parsed_data['labels'] = _custom_one_hot_vector(labels_doc_dict_list)
    return parsed_data


def _parse_twitter(file_path, text_processor):
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
    with open(file_path, "r") as file1:
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

    def __init__(self, file_path=None, dataset_info=None, graph_path=None):
        """
        Initializes the loader
        Args:
            file_path: Path to the file containing the dataset.
            dataset_info: Dictionary consisting of two fields 'dataset_name', 'num_classes'
            graph_path: path to the bin file containing the saved DGL graph
        """
        assert (file_path is not None and dataset_info is not None) or (
                graph_path is not None and dataset_info is not None), \
            "Either file_path and dataset_info should be specified or graph_path should be specified"

        if graph_path is None:
            assert file_path is not None, file_path + "does not exist!"
            self.file_path = file_path
            self.text_processor = TextProcessing()
            df = self.get_dataset_df()
            token_graph_ob = DGLGraph(df)
            self.graphs, labels_dict = token_graph_ob.create_instance_dgl_graphs()
            # TODO Take user input where to take save graphs
            # TODO dataset-name_[train-1]_graph.bin
            # TODO dataset-name_labels_ohv.bin
            # TODO use pathlib library to check validity of paths
            token_graph_ob.save_graphs("/home/abhi/Desktop/gcn/output/graph.bin", self.graphs, labels_dict)
            self.labels = labels_dict["glabel"]
            # TODO Put user inputted location here
            logging.info("Graph generated and stored at /home/abhi/Desktop/gcn/output/graph.bin")
        else:
            # TODO read the label id to word mapping from the file saved while generating labels
            self.graphs, self.labels = graph_utils.load_dgl_graphs(graph_path)

        self.dataset_info = dataset_info
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
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]

    def get_dataset_df(self):
        """
        Returns pandas dataframe
        """
        dataset_name = self.dataset_info["name"]
        if(dataset_name == "Twitter"):
            return _parse_twitter(self.file_path, self.text_processor)
        elif(dataset_name == "SemEval"):
            return _parse_sem_eval(self.file_path, self.text_processor)
        else:
            logging.error(
                "{} dataset not yet supported".format(self.dataset_name))
