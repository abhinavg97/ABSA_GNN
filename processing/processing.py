import spacy

from spacy.lang.en import English
from xml.etree import ElementTree as ET
from collections import Counter
import pandas as pd
import networkx as nx
import dgl
from spacy.vocab import Vocab
import torch
from spacy.vectors import Vectors

class Processing:

  def __init__(self):
    pass

  def parse_sem_eval(self, file_name):
    tree = ET.parse(file_name)
    root = tree.getroot() 
    data_row = []
    for review in root.findall('Review'):
      for sentences in review:
        for sentence in sentences:
          temp_row = ['lorem ipsum', []]
          temp_row[0]= sentence.find('text').text
          for opinions in sentence.findall('Opinions'):
            for opinion in opinions:
              polarity = opinion.get('polarity')
              if(polarity == 'positive'):
                temp_row[1] += [1]
              else:
                temp_row[1] += [-1]
          data_row += [temp_row]
    parsed_data = pd.DataFrame(data_row, columns = ['text', 'label'])  
    return parsed_data 

  def parse_twitter(self, file_name):    
    count = 0
    data_row = []
    with open(file_name, "r") as file1:
      for line in file1:
        stripped_line = line.strip()
        if count%3 == 0:
          temp_row = ['lorem ipsum', 0]
          temp_row[0] = stripped_line
        elif count%3 == 2:
          temp_row[1] = int(stripped_line)
          data_row += [temp_row]
        count += 1
    parsed_data = pd.DataFrame(data_row, columns = ['text', 'label'])
    return parsed_data        

  def networkx_graph(self, text):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                          '{0}'.format(child.lower_)))
    graph = nx.Graph(edges)
    return graph

  def __visulaize_dependancy_tree(self, doc):
    from pathlib import Path
    svg = spacy.displacy.render(doc, style='dep', jupyter=False)
    output_path = Path("/home/abhi/Desktop/temp.svg")
    output_path.open("w", encoding="utf-8").write(svg)

  def __visualize_dgl_graph_as_networkx(self, graph):
    graph = graph.to_networkx()
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, with_labels=False, node_size=10,
            node_color=[[.5, .5, .5]], arrowsize=4)

  def dgl_graph(self, text, index):
    graph = self.networkx_graph(text)
    g = dgl.DGLGraph()
    g.from_networkx(graph)

    # add edge weights and node weights
    vocab = self.create_vocab(text)

    # g.ndata['feat'] = 1

  def create_vocab(self, text):

    nlp = spacy.load("en_core_web_lg")
    tokens = nlp(text)
    words = {}
    vocab = Vocab()

    for token in tokens:
      if token.text not in words:
        words[token.text] = 1
        vocab.set_vector(token.text, token.vector)
    return vocab