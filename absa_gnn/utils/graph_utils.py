import networkx as nx
import matplotlib.pyplot as plt
from dgl.data.utils import save_graphs, load_graphs


def save_dgl_graphs(path, graphs, labels_dict=None):
    save_graphs(path, graphs, labels_dict)


def load_dgl_graphs(path, idx_list=None):
    graphs, labels_dict = load_graphs(path, idx_list)

    if len(labels_dict) != 0:
        labels_tensor_list = labels_dict["glabel"]
        labels_list = []
        for label in labels_tensor_list:
            labels_list += [label.tolist()]
    else:
        labels_list = None

    return graphs, labels_list


def visualize_dgl_graph_as_networkx(graph):
    graph = graph.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(graph)
    # pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, with_label=True)
    plt.show()
