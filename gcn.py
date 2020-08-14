import spacy
from nltk.tree import Tree
import pandas as pd
from processing import processing

class GCN:

  def __init__(self):
    self.en_nlp = spacy.load("en")
    self.data = pd.DataFrame({'text': [], 'label': []}) 
    self.processor = processing.Processing()

  def nltk_spacy_tree(self, sent):
      """
      Visualize the SpaCy dependency tree with nltk.tree
      """
      doc = self.en_nlp(sent)
      def token_format(token):
          return "_".join([token.orth_, token.tag_, token.dep_])

      def to_nltk_tree(node):
          if node.n_lefts + node.n_rights > 0:
              return Tree(token_format(node),
                         [to_nltk_tree(child) 
                          for child in node.children]
                     )
          else:
              return token_format(node)

      tree = [to_nltk_tree(sent.root) for sent in doc.sents]
      # The first item in the list is the full tree
      tree[0].draw()
   
  def run_gcn(self, file_name, dataset_name=None):
    if dataset_name == "twitter": 
      data = self.processor.parse_twitter(file_name)
    elif dataset_name == "semEval":
      data = self.processor.parse_sem_eval(file_name)
    else:
      print("Please input valid dataset_name: [twitter, semEval]")

    for index, item in data.iterrows():
      self.processor.dgl_graph(item[0], index)
      exit(0)
      self.nltk_spacy_tree(item[0])
  
gcn = GCN()
# gcn.run_sem_eval('SemEval16_gold_Laptops/EN_LAPT_SB1_TEST_.xml.gold') 
gcn.run_gcn('Twitter-acl-14-short-data/train.txt', dataset_name="twitter")
