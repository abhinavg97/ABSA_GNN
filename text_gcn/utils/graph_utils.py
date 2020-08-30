import networkx as nx
from nltk.tree import Tree
import matplotlib.pyplot as plt
from dgl.data.utils import save_graphs, load_graphs
import spacy


def visulaize_dependancy_tree(doc):
    from pathlib import Path
    svg = spacy.displacy.render(doc, style='dep', jupyter=False)
    output_path = Path("/home/abhi/Desktop/temp.svg")
    output_path.open("w", encoding="utf-8").write(svg)


def save_dgl_graphs(path, graphs, labels_dict=None):
    save_graphs(path, graphs, labels_dict)


def load_dgl_graphs(path, idx_list=None):
    load_graphs(path, idx_list)


def visualize_dgl_graph_as_networkx(graph):
    graph = graph.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(graph)
    # pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, with_label=True)
    plt.show()


def nltk_spacy_tree(sent):
    """
    Visualize the SpaCy dependency tree with nltk.tree
    """
    nlp = spacy.download("en_core_web_sm")
    doc = nlp(sent)

    def token_format(token):
        return "_".join([token.orth_, token.tag_, token.dep_])

    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(token_format(node), [to_nltk_tree(child)
                                             for child in node.children])
        else:
            return token_format(node)

    tree = [to_nltk_tree(sent.root) for sent in doc.sents]
    # The first item in the list is the full tree
    tree[0].draw()
