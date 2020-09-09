from text_gcn.models import GAT_Graph_Classifier
import pytorch_lightning as pl
# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('--TRAIN', type=bool, required=False, default=1, help='Switch to train on dataset')

# args = parser.parse_args()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = GAT_Graph_Classifier(in_dim=1, hidden_dim=4, num_heads=2,
                             n_classes=5)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Trainer Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


trainer = pl.Trainer(max_epochs=10)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer.fit(model)


# epochs = 5
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# trainer.train(epochs, optimizer)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculate the metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
