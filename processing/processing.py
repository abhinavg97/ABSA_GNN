import spacy

from spacy.lang.en import English
from xml.etree import ElementTree as ET
from collections import Counter
import pandas as pd
import networkx as nx
import dgl
import torch

class Processing:

  def __init__(self):
    self.id_to_vector = {}
    self.word_to_id = {}
    self.nlp = spacy.load("en_core_web_lg")
  
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


  def init(self, data):
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
        
  def dgl_graph(self, data):

    for _, item in data.iterrows():
      g = dgl.DGLGraph()
      tokens = self.nlp(item[0])
      embedding = []              # node embedding
      edges_sources = []           # edge data
      edges_dest = []             # edge data
      counter = 0                 # uniq ids for the tokens in the document
      uniq_ids = {}               # ids to map token to id for the dgl graph

      for token in tokens:
        if token.lower_ not in uniq_ids:
          uniq_ids[token.lower_] = counter
          embedding.append(token.vector)
          counter += 1

      for token in tokens:
        for child in token.children:
          edges_sources.append(uniq_ids[token.lower_])
          edges_dest.append(uniq_ids[child.lower_])

    # add edges and node embeddings to the graph
    g.add_nodes(counter)
    g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest))
    g.ndata['feat'] = torch.tensor(embedding)
