import nltk
import pandas as pd
import numpy as np
import spacy
from xml.etree import ElementTree as ET

import os
import glob


from sklearn.feature_extraction.text import TfidfVectorizer

from config import configuration as cfg
from logger.logger import logger

if cfg['training']['create_dataset'] and not cfg['DEBUG']:
    nlp = spacy.load("en_core_web_lg")


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
    with open("merged_" + cfg["data"]["dataset"]["name"] + ".xml", "wb") as f:
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


def print_dataframe_statistics(df, label_to_id):

    labels = np.array(df['labels'].tolist())
    labels_frequency = []
    for j in range(len(labels[0])):
        labels_frequency.append(0)
        for i in range(len(labels)):
            if labels[i][j] != -2:
                labels_frequency[j] += 1

    id_to_labels = {value: key for key, value in label_to_id.items()}

    id_to_label_list = [id_to_labels[index] for index in range(len(label_to_id))]
    labels_frequency_df = pd.DataFrame({'labels': id_to_label_list, 'frequency': labels_frequency})
    average_labels_per_sample = np.count_nonzero(labels != -2) / len(labels)
    average_samples_per_label = sum(labels_frequency)/len(labels_frequency)

    labels_frequency_df.to_csv(cfg["paths"]["data_root"] + cfg["data"]["dataset"]["name"] + "_bag_of_words.csv")
    logger.info("Number of samples in the dataset {}".format(len(labels)))
    logger.info("Number of labels in the dataset {}".format(len(labels[0])))
    logger.info("Average number of  labels per sample {}".format(average_labels_per_sample))
    logger.info("Average number of samples per label {}".format(average_samples_per_label))
    logger.info("Label frequence data \n{}".format(labels_frequency_df.sort_values(by=['frequency'], ascending=False)))


def prune_dataset_df(df, label_to_id):
    """
    prune labels which belong to fewer than 3 samples
    remove samples which don't contain any labels
    """
    labels = np.array(df['labels'].tolist())

    labels_frequency = []

    to_prune = []
    for j in range(len(labels[0])):
        labels_frequency.append(0)
        for i in range(len(labels)):
            if labels[i][j] != -2:
                labels_frequency[j] += 1
        if labels_frequency[j] <= cfg["data"]["min_label_occurences"]:
            to_prune.append(j)

    labels = np.delete(labels, to_prune, axis=1)

    df['labels'] = labels.tolist()

    id_to_labels = {value: key for key, value in label_to_id.items()}

    for index_to_remove in to_prune:
        del label_to_id[id_to_labels[index_to_remove]]

    if len(to_prune):
        accumulate = 0
        for i in range(len(to_prune)-1):
            accumulate += 1

            for j in range(to_prune[i]+1, to_prune[i+1]):
                label_to_id[id_to_labels[j]] -= accumulate

        for j in range(to_prune[-1]+1, len(id_to_labels)):
            label_to_id[id_to_labels[j]] -= (accumulate + 1)

        df.drop(df[df['labels'].apply(lambda x: x == [-2]*len(labels[0]))].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df, label_to_id


def pmi(df):
    """[summary]
        https://towardsdatascience.com/collocations-in-nlp-using-nltk-library-2541002998db
        https://www.nltk.org/howto/collocations.html
    Returns:
        list_of_tuples: tuples containing word pair and their associated pmi
    """

    bigram_measures = nltk.collocations.BigramAssocMeasures()

    tokenized = []

    for _, item in df.iterrows():
        tokenized += token_list(item[1])

    # Bigrams
    finder = nltk.collocations.BigramCollocationFinder.from_words(tokenized)
    # only bigrams that appear 3+ times
    # finder.apply_freq_filter(3)

    # return the 10 n-grams with the highest PMI
    # print (finder.nbest(bigram_measures.likelihood_ratio, 10))

    list_of_tuples = finder.score_ngrams(bigram_measures.pmi)
    return list_of_tuples


def iou(label_ohv1, label_ohv2):
    """
    Calculate the ious given two lists
    Args:
        l1: list1 containing one hot vector of labels
        l2: list2 containing one hot vectot of labels

    Returns:
        iou_score: IOU for the two labels lists
    """

    union = len(label_ohv1)
    intersection = 0
    for i, label1 in enumerate(label_ohv1):
        if label1 != 0 and label_ohv2[i] != 0:
            intersection += 1

    iou_score = intersection/union
    return iou_score


def tf_idf(df, vocab):
    """[summary]
    https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    Args:
        docs ([type]): [description]
    """

    docs = []
    for __, item in df.iterrows():
        docs += [item[1]]

    vectorizer = TfidfVectorizer(tokenizer=token_list, lowercase=False, vocabulary=vocab)
    vectors = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    tfidf_matrix = pd.DataFrame(denselist, columns=feature_names)
    return tfidf_matrix


def token_list(text):
    """
    Tokenizes text into individual tokens
    Returns a list of tokens from the text
    """
    tokens = []

    doc = nlp(text)
    for token in doc:
        tokens += [token.text]

    return tokens


def get_labels(df):
    """
    Fetches labels for the documents in the dataframe
    Args:
        df: dataframe to fetch the labels from
    """
    labels = []
    for index, doc in df.iterrows():
        labels += [doc[2]]
    return labels


if __name__ == "__main__":

    merge_semEval_14_type("../../data/SemEval14/")
