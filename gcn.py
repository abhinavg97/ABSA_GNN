
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
   
  def run_sem_eval(self, file_name):
    data = self.processor.parse_sem_eval(file_name)
    for _, item in data.iterrows():
      tokens = self.processor.tokenize(item[0])
      vocab = self.processor.create_vocab(tokens)
      self.nltk_spacy_tree(item[0])

  def run_twitter(self, file_name):
    data = self.processor.parse_twitter(file_name)
    for _, item in data.iterrows():
      tokens = self.processor.tokenize(item[0])
      vocab = self.processor.create_vocab(tokens)
      self.nltk_spacy_tree(item[0])
  
gcn = GCN()
# gcn.run_sem_eval('SemEval16_gold_Laptops/EN_LAPT_SB1_TEST_.xml.gold') 
gcn.run_twitter('Twitter-acl-14-short-data/train.txt')
