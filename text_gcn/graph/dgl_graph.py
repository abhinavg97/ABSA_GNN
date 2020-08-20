import spacy
import dgl
import torch
import numpy as np
from ..utils import graph_utils
from ..utils import utils


class DGLGraph(object):
    """
    Creates a DGL graph with training and testing functionality
    """

    def __init__(self, data):
        # counter is the variable storing the total number of docs + tokens
        self.total_nodes = 0
        self.id_to_vector = {}
        self.word_to_id = {}
        self.nlp = spacy.load("en_core_web_lg")
        words = {}
        counter = 0
        for _, item in data.iterrows():
            tokens = self.nlp(item[0])
            for token in tokens:
                if token.lower_ not in words:
                    words[token.lower_] = 1
                    self.id_to_vector[counter] = token.vector
                    self.word_to_id[token.lower_] = counter
                    counter += 1
                else:
                    words[token.lower_] = 1

        self.total_nodes = counter + data.shape[0]
        self.dataframe = data

    def create_individual_dgl_graphs(self):
        """
        Constructs individual DGL graphs for each of the data samples
        Returns:
            graphs: An array containing DGL graphs
        """
        graphs = []
        for _, item in self.dataframe.iterrows():
            graphs += [self.create_single_dgl_graph(item[0])]
        return graphs

    def visualize_dgl_graph(self, graph):
        """
        visualize single dgl graph
        Args:
            graph: dgl graph
        """
        graph_utils._visualize_dgl_graph_as_networkx(graph)

    def save_graphs(self, graphs):
        graph_utils._save_graphs("../../bin/graph.bin", graphs)

    def create_single_dgl_graph(self, text):
        """
        Create a single DGL graph
        Args:
            text: Input data in string format

        Returns:
            DGL Graph: DGL graph for the input text
        """
        g = dgl.DGLGraph()
        tokens = self.nlp(text)
        embedding = []              # node embedding
        edges_sources = []          # edge data
        edges_dest = []             # edge data
        counter = 0                 # uniq ids for the tokens in the document
        uniq_ids = {}               # ids to map token to id for the dgl graph
        token_ids = []              # global unique ids for the tokens

        for token in tokens:
            if token.lower_ not in uniq_ids:
                uniq_ids[token.lower_] = counter
                embedding.append(token.vector)
                counter += 1
                token_ids += [self.word_to_id[token.lower_]]

        for token in tokens:
            for child in token.children:
                edges_sources.append(uniq_ids[token.lower_])
                edges_dest.append(uniq_ids[child.lower_])

        # add edges and node embeddings to the graph
        g.add_nodes(len(uniq_ids.keys()))
        g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest))
        g.ndata['feat'] = torch.tensor(embedding)
        # add token id attribute to node
        g.ndata['token_id'] = torch.tensor(token_ids)
        return g

    def create_complete_dgl_graph(self):
        """
        Creates a complete dgl graph tokens and documents as nodes
        """
        g = dgl.DGLGraph()
        g.add_nodes(self.total_nodes)

        # add node data for vocab nodes
        for id in self.word_to_id.values():
            g.nodes[id].data['id'] = np.array([id])
            g.nodes[id].data['feat'] = np.array([self.id_to_vector[id]])

        pmi = utils.pmi(self.dataframe)
        # add edges and edge data betweem vocab words in the dgl graph
        for tuples in pmi:
            word_pair = tuples[0]
            pmi_score = tuples[1]
            word1 = word_pair[0]
            word2 = word_pair[1]
            word1_id = self.word_to_id[word1]
            word2_id = self.word_to_id[word2]
            g.add_edge(word1_id, word2_id)
            g.edges[word1_id, word2_id].data['weight'] = np.array([pmi_score])

        # add edges and edge data between documents
        for index1, doc1 in self.dataframe.iterrows():
            for index2, doc2 in self.dataframe.iterrows():
                if index1 != index2:
                    doc1_id = len(self.word_to_id.keys()) + index1
                    doc2_id = len(self.word_to_id.keys()) + index2
                    weight = utils.iou(doc1[1], doc2[1])
                    g.add_edge(doc1_id, doc2_id)
                    g.edges[doc1_id, doc2_id].data['weight'] = np.array([weight])
                    g.nodes[doc1_id].data['id'] = np.array([doc1_id])

        tf_idf_df = utils.tf_idf(self.dataframe)
        # add edges and edge data between word and documents
        for index, doc_row in tf_idf_df.iterrows():

            doc_id = len(self.word_to_id.keys()) + index

            for word, tf_idf_value in doc_row.items():
                word_id = self.word_to_id[word]
                g.add_edge(doc_id, word_id)
                g.edges[doc_id, word_id].data['weight'] = np.array([tf_idf_value])

        return g
