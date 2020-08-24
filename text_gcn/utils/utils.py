from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import itertools
import nltk
import pandas as pd
import spacy


nlp = spacy.load("en_core_web_lg")


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
        tokenized += token_list(item[0])

    # Bigrams
    finder = nltk.collocations.BigramCollocationFinder.from_words(tokenized)
    # only bigrams that appear 3+ times
    # finder.apply_freq_filter(3)

    # return the 10 n-grams with the highest PMI
    # print (finder.nbest(bigram_measures.likelihood_ratio, 10))

    list_of_tuples = finder.score_ngrams(bigram_measures.pmi)
    return list_of_tuples


def iou(l1, l2):
    """
    Calculate the ious given two lists
    Args:
        l1: list1 containing labels
        l2: list2 containing labels

    Returns:
        iou_score: IOU for the two labels lists
    """
    if(len(l1) == 0 or len(l2) == 0):
        return 0.0

    union_len = len(l1) + len(l2)
    intersection = []
    for x in l1:
        if x in l2:
            intersection.append(x)
            l2.remove(x)
    union_len -= len(intersection)
    iou_score = len(intersection)/union_len

    return iou_score


def tf_idf(df, vocab):
    """[summary]
    https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    Args:
        docs ([type]): [description]
    """

    docs = []
    for _, item in df.iterrows():
        docs += [item[0]]

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
        labels += [doc[1]]
    return labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/abhi/Desktop/gcn/text_gcn/loaders/")

    from loader_gcn import GCNLoader

    train_loader = GCNLoader(
        "../../data/SemEval16_gold_Laptops/train.txt", dataset_name="SemEval")

    df = train_loader.get_dataframe()

    tf_df = tf_idf(df)
    # print(tf_df)

    for index, row in tf_df.iterrows():
        for i, v in row.items():
            print(i, v)
        # print(type(item))
