import spacy

from spacy.lang.en import English
from xml.etree import ElementTree as ET
from collections import Counter
import pandas as pd

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

  def tokenize(self, text):
    tokens = []
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
      tokens += [token.text]
    return tokens

  def create_vocab(self, tokens):
    cnt = Counter()
    for word in tokens:
      cnt[word] += 1
    return cnt