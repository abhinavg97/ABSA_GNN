import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from text_gcn.models import GAT_Graph_Classifier
from text_gcn.loaders import GraphDataModule

from config import configuration as cfg
# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('--TRAIN', type=bool, required=False, default=1, help='Switch to train on dataset')

# args = parser.parse_args()

# TODO write arguement parser and pass the required params to dm
dm = GraphDataModule()

# OPTIONAL

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logger initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# logger = pl.loggers.TensorBoardLogger("lightning_logs", name=cfg['data']['dataset']['name'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = GAT_Graph_Classifier(in_dim=cfg['model']['in_dim'], hidden_dim=cfg['model']['hidden_dim'],
                             num_heads=cfg['model']['num_heads'], n_classes=dm.num_classes)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Trainer Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


early_stop_callback = EarlyStopping(
    monitor='val_f1_score',
    min_delta=cfg['training']['early_stopping_delta'],
    patience=cfg['training']['early_stopping_patience'],
    verbose=True,
    mode='max'
    )

cuda_available = torch.cuda.is_available()

trainer = pl.Trainer(max_epochs=cfg['training']['epochs'], log_every_n_steps=50, auto_scale_batch_size='binsearch',
                     gpus=-1, auto_select_gpus=cuda_available, accelerator='ddp2', auto_lr_find=True, fast_dev_run=False,
                     num_sanity_val_steps=0, callbacks=[early_stop_callback])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer.fit(model, dm)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Test your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer.test(datamodule=dm)
