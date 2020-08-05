import spacy

from xml.etree import ElementTree as ET
from nltk.tree import Tree



class GCN:

  def __init__(self):
    self.en_nlp = spacy.load("en")
  
  def parse_sem_eval(self):
    tree = ET.parse(self.file_name)
    root = tree.getroot() 
    parsed_data = []   
    for review in root.findall('Review'):
      for sentences in review:
        for sentence in sentences:
          parsed_data.append((sentence.find('text').text))
    return parsed_data 

  def parse_twitter(self):
    parsed_data = []
    
    count = 0
    with open(self.file_name, "r") as file1:
      for line in file1:
        stripped_line = line.strip()
        if count%3 == 0:
          parsed_data.append(stripped_line)
        count += 1
    return parsed_data        
    

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
    self.file_name = file_name
    data = self.parse_sem_eval()
    for text in data:
      self.nltk_spacy_tree(text)

  def run_twitter(self, file_name):
    self.file_name = file_name
    data = self.parse_twitter()
    for text in data:
      self.nltk_spacy_tree(text)
    

gcn = GCN()
#gcn.run_sem_eval('SemEval16_gold_Laptops/EN_LAPT_SB1_TEST_.xml.gold') 
gcn.run_twitter('Twitter-acl-14-short-data/train.txt')
