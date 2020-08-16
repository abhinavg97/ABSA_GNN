from processing import processing

class Graph:

  def __init__(self):
    self.processor = processing.Processing()
   
  def run_gcn(self, file_name, dataset_name=None):
    if dataset_name == "twitter": 
      data = self.processor.parse_twitter(file_name)
    elif dataset_name == "semEval":
      data = self.processor.parse_sem_eval(file_name)
    else:
      print("Please input valid dataset_name: [twitter, semEval]")
    
    # self.processor.init(data)
    self.processor.dgl_graph(data)
    exit(0)
    for _, item in data.iterrows():
      print(item[0])
      self.processor.nltk_spacy_tree(item[0])
  
gcn = Graph()
# gcn.run_sem_eval('SemEval16_gold_Laptops/EN_LAPT_SB1_TEST_.xml.gold') 
gcn.run_gcn('Twitter-acl-14-short-data/train.txt', dataset_name="twitter")