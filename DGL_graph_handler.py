from text_gcn.models import GAT_Graph_Classifier
import pytorch_lightning as pl
from text_gcn.loaders import GraphDataModule
from config import configuration as cfg
# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('--TRAIN', type=bool, required=False, default=1, help='Switch to train on dataset')

# args = parser.parse_args()

# TODO write arguement parser and pass the required params to dm
dm = GraphDataModule()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = GAT_Graph_Classifier(in_dim=cfg['model']['in_dim'], hidden_dim=cfg['model']['hidden_dim'],
                             num_heads=cfg['model']['num_heads'], n_classes=dm.num_classes)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Trainer Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer = pl.Trainer(max_epochs=cfg['training']['max_epochs'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer.fit(model, dm)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Test your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer.test(datamodule=dm)
