import nltk
import pandas as pd
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

from config import configuration as cfg
from logger.logger import logger


if cfg['training']['create_dataset'] or not cfg['DEBUG']:
    nlp = spacy.load("en_core_web_lg")


def split_data(sample_keys, labels, test_size, stratified, random_state=0):
    """
    Splits the data into 2 parts with optional stratified splitting
    Args:
        sample_keys: List of index values for x
        labels: List of one hot vectors
        test_size: test size of the split
        stratified: Toggle to switch on the stratifies split
        random_state: Defaults to 0.

    Returns:
        X indices of the split and corresponding y values
    """
    if stratified is False:
        x_split, y_split, x_labels, y_labels = train_test_split(sample_keys, labels, test_size=test_size, random_state=random_state)
    else:
        x_split, x_labels, y_split, y_labels = iterative_train_test_split(sample_keys, labels, test_size=test_size)

    x_labels = x_labels.todense().tolist()
    y_labels = y_labels.todense().tolist()
    x_split = x_split.todense().tolist()
    y_split = y_split.todense().tolist()

    return x_labels, y_labels, x_split, y_split


def print_dataframe_statistics(df, label_text_to_label_id):

    labels = np.array(df['labels'].tolist())
    labels_frequency = []
    for j in range(len(labels[0])):
        labels_frequency.append(0)
        for i in range(len(labels)):
            if labels[i][j] != -2:
                labels_frequency[j] += 1

    label_id_to_label_text = {value: key for key, value in label_text_to_label_id.items()}

    id_to_label_list = [label_id_to_label_text[index] for index in range(len(label_text_to_label_id))]
    labels_frequency_df = pd.DataFrame({'labels': id_to_label_list, 'frequency': labels_frequency})
    average_labels_per_sample = np.count_nonzero(labels != -2) / len(labels)
    average_samples_per_label = sum(labels_frequency)/len(labels_frequency)

    labels_frequency_df.to_csv(cfg['paths']['data_root'] + cfg['data']['dataset']['name'] + "_bag_of_words.csv")
    logger.info("Number of samples in the dataset {}".format(len(labels)))
    logger.info("Number of labels in the dataset {}".format(len(labels[0])))
    logger.info("Average number of  labels per sample {}".format(average_labels_per_sample))
    logger.info("Average number of samples per label {}".format(average_samples_per_label))
    logger.info("Label frequence data \n{}".format(labels_frequency_df.sort_values(by=['frequency'], ascending=False)))


def prune_dataset_df(df, label_text_to_label_id):
    """
    1. prune labels which belong to fewer than 3 samples
    2. remove samples which don't contain any labels
    """
    labels = np.array(df['labels'].tolist())

    labels_frequency = []

    to_prune = []
    for j in range(len(labels[0])):
        labels_frequency.append(0)
        for i in range(len(labels)):
            if labels[i][j] != -2:
                labels_frequency[j] += 1
        if labels_frequency[j] <= cfg['data']['min_label_occurences']:
            to_prune.append(j)

    labels = np.delete(labels, to_prune, axis=1)

    df['labels'] = labels.tolist()

    label_id_to_label_text = {value: key for key, value in label_text_to_label_id.items()}

    for index_to_remove in to_prune:
        del label_text_to_label_id[label_id_to_label_text[index_to_remove]]

    if len(to_prune):
        accumulate = 0
        for i in range(len(to_prune)-1):
            accumulate += 1

            for j in range(to_prune[i]+1, to_prune[i+1]):
                label_text_to_label_id[label_id_to_label_text[j]] -= accumulate

        for j in range(to_prune[-1]+1, len(label_id_to_label_text)):
            label_text_to_label_id[label_id_to_label_text[j]] -= (accumulate + 1)

        df.drop(df[df['labels'].apply(lambda x: x == [-2]*len(labels[0]))].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df, label_text_to_label_id


def pmi(df):
    """[summary]
        https://towardsdatascience.com/collocations-in-nlp-using-nltk-library-2541002998db
        https://www.nltk.org/howto/collocations.html
    Returns:
        list_of_tuples: tuples containing word pair and their associated pmi
    """

    bigram_measures = nltk.collocations.BigramAssocMeasures()

    tokenized = []

    for text in df['text'].tolist():
        tokenized += token_list(text)

    # Bigrams
    finder = nltk.collocations.BigramCollocationFinder.from_words(tokenized)
    # only bigrams that appear 3+ times
    # finder.apply_freq_filter(3)

    # return the 10 n-grams with the highest PMI
    # print (finder.nbest(bigram_measures.likelihood_ratio, 10))

    list_of_tuples = finder.score_ngrams(bigram_measures.pmi)
    return list_of_tuples


def iou(label_mhv1, label_mhv2):
    """
    Calculate the ious given two lists
    Args:
        l1: list1 containing one hot vector of labels
        l2: list2 containing one hot vectot of labels

    Returns:
        iou_score: IOU for the two labels lists
    """

    union = len(label_mhv1)
    intersection = 0
    for i, label1 in enumerate(label_mhv1):
        if label1 != -2 and label_mhv2[i] != -2:
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
    for text in df['text'].tolist():
        docs += [text]

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
