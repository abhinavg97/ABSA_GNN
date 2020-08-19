from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import itertools
from scipy.sparse import csr_matrix
import nltk
from nltk.collocations import *


def pmi(docs):
    """[summary]
        https://towardsdatascience.com/collocations-in-nlp-using-nltk-library-2541002998db
        https://www.nltk.org/howto/collocations.html
    Returns:
        list_of_tuples: tuples containing word pair and their associated pmi
    """
    separator = ', '
    text = separator.join(docs)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    tokens = nltk.wordpunct_tokenize(text)
    # Bigrams
    finder = BigramCollocationFinder.from_words(tokens)
    # only bigrams that appear 3+ times
    # finder.apply_freq_filter(3)

    # return the 10 n-grams with the highest PMI
    # print (finder.nbest(bigram_measures.likelihood_ratio, 10))

    list_of_tuples = finder.score_ngrams(bigram_measures.pmi)
    return list_of_tuples


def similarity_pairwise_words():
    pass


def similarity_pairwise_documents(doc1, doc2):
    text1 = doc1[text]
    text2 = doc2[text]
    labels1 = doc1[labels]
    labels2 = doc2[labels]

    def iou(l1, l2):
        union_len = len(l1)+len(l2)
        intersection = []
        for x in l1:
            if x in l2:
                intersection.append(x)
                l2.remove(x)
        union_len -= len(intersection)
        return len(intersection)/union_len

    return iou(labels1, labels2)


def tf_idf(docs):
    """[summary]
    https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    Args:
        docs ([type]): [description]
    """

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    tfidf_matrix = pd.DataFrame(denselist, columns=feature_names)
    return tfidf_matrix


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/abhi/Desktop/gcn/Processing/")

    from processing import Processing
    processor = Processing()
    data = processor.parse_twitter("../Twitter-acl-14-short-data/train.txt")

    docs = []

    for _, item in data.iterrows():
        docs += [item[0]]
    pmi(docs)
