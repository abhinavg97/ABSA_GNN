import re
import os
import glob
import spacy
import string
import pandas as pd
from xml.etree import ElementTree as ET

from config import configuration as cfg
# TODO clean sem_eval_14 dataset -> labels like windows_xp to operating system

nlp = spacy.load('en_core_web_sm')


def _clean_term(label):
    label = label.lower()
    all_stopwords = nlp.Defaults.stop_words
    label_tokens = nlp(label)
    label = " ".join([str(word) for word in label_tokens if word not in all_stopwords])
    label = re.sub('[^A-Za-z0-9 ]+', '', label)
    label = label[:30]
    return label


def _multi_hot_vector(labels_doc_dict_list):
    """
    Creates multi hot vector for the labels
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
    multi_hot_vector = [zero_vector[:] for i in range(len(labels_doc_dict_list))]

    for i, labels in enumerate(labels_doc_dict_list):
        for label in labels.keys():
            multi_hot_vector[i][label_text_to_label_id[label]] = labels[label]

    return multi_hot_vector, label_text_to_label_id


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
    parsed_data['labels'], label_text_to_label_id = _multi_hot_vector(labels_doc_dict_list)
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
    parsed_data['labels'], label_text_to_label_id = _multi_hot_vector(labels_doc_dict_list)
    return parsed_data, label_text_to_label_id


def _parse_text_gcn_type(dataset_path, text_processor=None):
    """
    Function to parse text gcn dataframes given in the data directory
    Args:
    dataset_path: path of the dataframe to be parsed
    Returns:
        parsed_data: pandas dataframe for the parsed
        data containing labels and text
    """
    parsed_data = pd.read_csv(dataset_path, index_col=0)
    labels = parsed_data['labels']
    labels_doc_dict_list = list(map(lambda label: {label: 1}, labels))
    parsed_data['labels'], label_text_to_label_id = _multi_hot_vector(labels_doc_dict_list)
    parsed_data['id'] = list(range(len(labels)))

    return parsed_data, label_text_to_label_id


def _parse_twitter(dataset_path, text_processor=None):
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


def merge_semEval_16_type(folder):
    """
    Takes in a folder and merges all the xml files in it
    Merge SemEval16 type xml files via this function
    Args:
        folder: folder path to merge files
    """
    xml_files = glob.glob(folder+"/*.xml")

    node = None
    for xmlFile in xml_files:
        tree = ET.parse(xmlFile)
        root = tree.getroot()
        if node is None:
            node = root
        else:
            elements = root.findall("./Review")
            for element in elements:
                node[1].append(element)
    with open("merged_" + cfg['data']['dataset']['name'] + ".xml", "wb") as f:
        f.write(ET.tostring(node))


def merge_semEval_14_type(folder):
    """
    Takes in a folder and merges all the xml files in it
    Merge SemEval14 type xml files via this function
    Make sure to delete any uneccesary xml files including any previously merged files
        from the directory before proceeding
    Args:
        folder: folder path to merge files
    """
    xml_files = glob.glob(os.path.join(os.path.dirname(__file__), folder)+"*.xml")

    node = None
    for xmlFile in xml_files:
        tree = ET.parse(xmlFile)
        root = tree.getroot()
        if node is None:
            node = root
        else:
            elements = root.findall("./sentence")
            for element in elements:
                node[1].append(element)

    with open(os.path.join(os.path.dirname(__file__), folder+"merged_files/"+"merged_" + "SemEval14" + ".xml"), "wb") as f:
        f.write(ET.tostring(node))
