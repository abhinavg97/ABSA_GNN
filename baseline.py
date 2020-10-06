import torch
import numpy as np
from scipy.sparse import lil_matrix
from simpletransformers.classification import MultiLabelClassificationModel

from text_gcn import utils
from text_gcn.loaders.graph_loader import GraphDataset
from text_gcn.metrics import class_wise_f1_scores, class_wise_precision_scores, class_wise_recall_scores, f1_scores_average

from logger.logger import logger
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

    gd = GraphDataset(label_text_to_label_id_path='',
                      dataframe_df_path='', dataset_path=cfg['paths']['data_root']+cfg['paths']['dataset'],
                      dataset_info=cfg['data']['dataset'], graph_path='')
    df, label_test_to_label_id = gd.get_dataset_df()

    x_df, y_df = split_data(df=df, test_size=0.3, stratified=True, random_state=1)

    num_classes = gd.num_classes
    return x_df, y_df, num_classes


train_df, eval_df, num_classes = read_data()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = MultiLabelClassificationModel('distilbert', 'distilbert-base-uncased-distilled-squad',
                                      use_cuda=False, num_labels=num_classes,
                                      args={'reprocess_input_data': True,
                                            'overwrite_output_dir': True,
                                            'num_train_epochs': 1})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.train_model(train_df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculate metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

labels = torch.Tensor(eval_df['labels'].tolist())

logger.info(f"class wise f1 scores: \n {class_wise_f1_scores(model_outputs, labels)}")
logger.info(f"class wise precision scores: \n {class_wise_precision_scores(model_outputs, labels)}")
logger.info(f"class wise recall scores: \n {class_wise_recall_scores(model_outputs, labels)}")
logger.info(f"f1 scores average: \n {f1_scores_average(model_outputs, labels)}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Use your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# predictions, raw_outputs = model.predict(['This thing is entirely different from the other thing. '])
