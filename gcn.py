
from text_gcn.graph import DGLGraph
from text_gcn.loaders import GCNLoader

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# self.processor.dgl_graph(data)
# for _, item in data.iterrows():
#     print(item[0])
# self.processor.nltk_spacy_tree(item[0])
# self.processor.large_dgl_graph(data)


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


# data argument
parser.add_argument('--data_file', type=str, required=True,
                    help="""File path for dataset. 
                    Must contain the dataset for
                    Twitter dataset or SemEval Dataset""")

parser.add_argument('--dataset_name', type=str, required=True,
                    help='Must be either Twitter or SemEval')

# parser.add_arguement('--output_dir', type=str, help='Output directory')


# parse the arguments
args = parser.parse_args()


train_loader = GCNLoader(args.data_file, args.dataset_name)

dataset = train_loader.get_dataframe()

gcn = DGLGraph(dataset)
graphs = gcn.create_individual_dgl_graphs()
gcn.visualize_dgl_graph(graphs[0])
# gcn.run_sem_eval('SemEval16_gold_Laptops/EN_LAPT_SB1_TEST_.xml.gold')
# gcn.run_gcn('Twitter-acl-14-short-data/train.txt', dataset_name="twitter")
