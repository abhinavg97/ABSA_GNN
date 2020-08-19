import spacy
import dgl
import torch
from ..utils import graph_utils


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
                if token.text not in words:
                    words[token.text] = 1
                    self.id_to_vector[counter] = token.vector
                    self.word_to_id[token.lower_] = counter
                    counter += 1
                else:
                    words[token.text] = 1

        self.total_nodes = counter + data.shape[0]
        self.data = data

    def create_complete_dgl_graph(self):
        pass

    def create_individual_dgl_graphs(self):
        """
        Constructs individual DGL graphs for each of the data samples
        Returns:
            graphs: An array containing DGL graphs
        """
        graphs = []
        for _, item in self.data.iterrows():
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
