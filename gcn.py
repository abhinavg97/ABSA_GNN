import spacy

from xml.etree import ElementTree as ET
from nltk.tree import Tree



class GCN:

  def __init__(self, file_name):
    self.file_name = file_name 
    self.en_nlp = spacy.load("en")
  
  def parse(self):
    tree = ET.parse(self.file_name)
    root = tree.getroot() 
    parsed_data = []   
    for review in root.findall('Review'):
      for sentences in review:
        for sentence in sentences:
          parsed_data.append((sentence.find('text').text))
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
   
  def run(self):
    data = self.parse()
    for text in data:
      self.nltk_spacy_tree(text)


gcn = GCN('SemEval16_gold_Laptops/EN_LAPT_SB1_TEST_.xml.gold')
gcn.run() 

