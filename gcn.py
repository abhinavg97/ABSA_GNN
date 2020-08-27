from text_gcn.graph import DGLGraph
from text_gcn.loaders import GCNLoader

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


# data argument
parser.add_argument('--data_file', type=str, required=True,
                    help="""File path for dataset.
                    Must contain the dataset for
                    Twitter dataset or SemEval Dataset""")

parser.add_argument('--dataset_name', type=str, required=True,
                    help='Must be either "Twitter" or "SemEval"')

# parser.add_arguement('--output_dir', type=str, help='Output directory')


# parse the arguments
args = parser.parse_args()


train_loader = GCNLoader(args.data_file, args.dataset_name)

dataset = train_loader.get_dataframe()

gcn = DGLGraph(dataset)
graphs = gcn.create_individual_dgl_graphs()
# graph = gcn.create_complete_dgl_graph()
gcn.visualize_dgl_graph(graphs[0])
gcn.save_graphs("/home/abhi/Desktop/gcn/output/graph.bin", graphs)
