import json
import torch
import numpy as np
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger

from ast import literal_eval
from scipy.sparse import lil_matrix
from simpletransformers.classification import MultiLabelClassificationModel

from text_gcn import utils
from text_gcn.metrics import class_wise_f1_scores, class_wise_precision_scores, class_wise_recall_scores, f1_scores_average

from config import configuration as cfg


def split_data(df, stratified=True, test_size=0.3, random_state=1):

    sample_keys = df.index.values
    sample_keys = lil_matrix(np.reshape(sample_keys, (len(sample_keys), -1)))

    labels = df['labels'].tolist()
    labels = list(map(lambda label_vec: list(map(lambda x: 0 if x == -2 else 1, label_vec)), labels))
    df['labels'] = labels
    labels = lil_matrix(np.array(labels))

    x_labels, y_labels, x_split, y_split = utils.split_data(sample_keys=sample_keys, labels=labels,
                                                            test_size=test_size, stratified=stratified,
                                                            random_state=random_state)

    x_split = np.reshape(x_split, (len(x_split), ))
    y_split = np.reshape(y_split, (len(y_split), ))

    x_df = df.loc[x_split]
    y_df = df.loc[y_split]

    return x_df, y_df


def read_data():

    df = pd.read_csv(cfg['paths']['data_root']+cfg['paths']['dataframe'], index_col=0)
    df['labels'] = list(map(lambda label_list: literal_eval(label_list), df['labels'].tolist()))

    x_df, y_df = split_data(df=df, test_size=0.3, stratified=True, random_state=1)
    num_classes = len(df['labels'][0])

    return x_df, y_df, num_classes


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Read data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

label_text_to_label_id_path = cfg['paths']['data_root'] + cfg['paths']['label_text_to_label_id']

with open(label_text_to_label_id_path, "r") as f:
    label_text_to_label_id = json.load(f)

label_id_to_label_text = {value: key for key, value in label_text_to_label_id.items()}


train_df, eval_df, num_classes = read_data()

train_df = train_df[:4]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logger initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

logger = TensorBoardLogger("lightning_logs", name="baseline")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = MultiLabelClassificationModel('distilbert', 'distilbert-base-uncased-distilled-squad',
                                      use_cuda=False, num_labels=num_classes,
                                      args={'reprocess_input_data': True,
                                            'overwrite_output_dir': True,
                                            'num_train_epochs': 1,
                                            'threshold': 0.5, 'tensorboard_dir': 'lightning_logs/baseline'})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.train_model(train_df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculate metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

labels = torch.Tensor(eval_df['labels'].tolist())

class_f1_scores = class_wise_f1_scores(model_outputs > 0.5, labels)
class_precision_scores = class_wise_precision_scores(model_outputs > 0.5, labels)
class_recall_scores = class_wise_recall_scores(model_outputs > 0.5, labels)
f1_scores_average = f1_scores_average(model_outputs > 0.5, labels)

class_f1_scores_dict = {label_id_to_label_text[i]: class_f1_scores[i] for i in range(len(label_id_to_label_text))}
class_recall_scores_dict = {label_id_to_label_text[i]: class_recall_scores[i] for i in range(len(label_id_to_label_text))}
class_precision_scores_dict = {label_id_to_label_text[i]: class_precision_scores[i] for i in range(len(label_id_to_label_text))}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

logger.experiment.add_scalars('test_class_f1_scores', class_f1_scores_dict, global_step=0)
logger.experiment.add_scalars('test_class_recall_scores', class_recall_scores_dict)
logger.experiment.add_scalars('test_class_precision_scores', class_precision_scores_dict)
logger.log_metrics(metrics={'avg_test_f1_score': f1_scores_average.item()})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Use your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# predictions, raw_outputs = model.predict(['This thing is entirely different from the other thing. '])
