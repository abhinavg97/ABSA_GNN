import json
import torch
import numpy as np
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger

from ast import literal_eval
from scipy.sparse import lil_matrix
from simpletransformers.classification import MultiLabelClassificationArgs
from simpletransformers.classification import MultiLabelClassificationModel

from acsa_gnn import utils
from acsa_gnn.metrics import class_wise_f1_scores, class_wise_precision_scores, class_wise_recall_scores,\
                             f1_score, accuracy_score, precision_score, recall_score

from config import configuration as cfg


def split_data(df, stratified=True, test_size=0.3, random_state=1, custom_one_hot=False):

    sample_keys = df.index.values
    sample_keys = lil_matrix(np.reshape(sample_keys, (len(sample_keys), -1)))

    labels = df['labels'].tolist()

    if custom_one_hot:
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

    train_val_df, test_df = split_data(df=df, test_size=cfg['data']['trainval_test_split'], stratified=True, random_state=1,
                                       custom_one_hot=True)

    train_df, val_df = split_data(df=train_val_df, test_size=cfg['data']['train_val_split'], stratified=True, random_state=1)

    num_classes = len(df['labels'][0])

    label_text_to_label_id_path = cfg['paths']['data_root'] + cfg['paths']['label_text_to_label_id']

    with open(label_text_to_label_id_path, "r") as f:
        label_text_to_label_id = json.load(f)

    label_id_to_label_text = {value: key for key, value in label_text_to_label_id.items()}

    return train_df, val_df, test_df, num_classes, label_id_to_label_text


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Read data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_df, val_df, test_df, num_classes, label_id_to_label_text = read_data()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cuda_available = torch.cuda.is_available()
n_gpu = torch.cuda.device_count()


model_args = MultiLabelClassificationArgs()

# model_args.save_model_every_epoch = True
# model_args.no_save = False
model_args.n_gpu = n_gpu
model_args.dataloader_num_workers = cfg['hardware']['num_workers']
model_args.no_cache = True
model_args.save_eval_checkpoints = False
model_args.save_optimizer_and_scheduler = False
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = cfg['training']['epochs']
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_each_epoch = True
model_args.use_early_stopping = True
model_args.early_stopping_consider_epochs = True
model_args.early_stopping_metric = "avg_val_f1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = cfg['training']['early_stopping_patience']
model_args.early_stopping_delta = cfg['training']['early_stopping_delta']
model_args.train_batch_size = cfg['training']['train_batch_size']
model_args.eval_batch_size = cfg['training']['val_batch_size']
model_args.threshold = cfg['training']['threshold']
model_args.tensorboard_dir = 'lightning_logs/baseline'
model_args.config = {'id2label': label_id_to_label_text}
model_args.manual_seed = cfg['training']['seed']

model = MultiLabelClassificationModel('distilbert', 'distilbert-base-uncased-distilled-squad',
                                      use_cuda=cuda_available, num_labels=num_classes, args=model_args)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.train_model(train_df, eval_df=val_df, output_dir="outputs", avg_val_accuracy_score=accuracy_score,
                  avg_val_f1_score=f1_score, avg_val_precision_score=precision_score, avg_val_recall_score=recall_score,
                  val_class_wise_f1_scores=class_wise_f1_scores, val_class_wise_precision_scores=class_wise_precision_scores,
                  val_class_wise_recall_scores=class_wise_recall_scores)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

result, model_outputs, wrong_predictions = model.eval_model(test_df, output_dir="outputs", f1_score=f1_score)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculate metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

labels = torch.Tensor(test_df['labels'].tolist())
train_val_metrics = pd.read_csv('outputs/training_progress_scores.csv', index_col=0)

avg_test_f1_score = f1_score(labels, model_outputs)
avg_test_precision_score = precision_score(labels, model_outputs)
avg_test_recall_score = recall_score(labels, model_outputs)
avg_test_accuracy_score = accuracy_score(labels, model_outputs)

test_class_f1_scores = class_wise_f1_scores(labels, model_outputs)
test_class_precision_scores = class_wise_precision_scores(labels, model_outputs)
test_class_recall_scores = class_wise_recall_scores(labels, model_outputs)

test_class_f1_scores_dict = {label_id_to_label_text[i]: test_class_f1_scores[i] for i in range(len(label_id_to_label_text))}
test_class_recall_scores_dict = {label_id_to_label_text[i]: test_class_recall_scores[i] for i in range(len(label_id_to_label_text))}
test_class_precision_scores_dict = {label_id_to_label_text[i]: test_class_precision_scores[i] for i in range(len(label_id_to_label_text))}

avg_train_loss = train_val_metrics['train_loss'].tolist()
avg_val_loss = train_val_metrics['eval_loss'].tolist()
avg_val_f1_score = train_val_metrics['avg_val_f1_score'].tolist()
avg_val_precision_score = train_val_metrics['avg_val_precision_score'].tolist()
avg_val_recall_score = train_val_metrics['avg_val_recall_score'].tolist()
avg_val_accuracy_score = train_val_metrics['avg_val_accuracy_score'].tolist()

val_class_f1_scores_list = train_val_metrics['val_class_wise_f1_scores'].tolist()
val_class_precision_scores_list = train_val_metrics['val_class_wise_precision_scores'].tolist()
val_class_recall_scores_list = train_val_metrics['val_class_wise_recall_scores'].tolist()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logger initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

logger = TensorBoardLogger("lightning_logs", name="baseline")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

epochs = len(avg_train_loss)

for epoch in range(epochs):

    val_class_f1_scores = literal_eval(val_class_f1_scores_list[epoch])
    val_class_precision_scores = literal_eval(val_class_precision_scores_list[epoch])
    val_class_recall_scores = literal_eval(val_class_recall_scores_list[epoch])

    val_class_f1_scores_dict = {label_id_to_label_text[i]: val_class_f1_scores[i] for i in range(len(label_id_to_label_text))}
    val_class_recall_scores_dict = {label_id_to_label_text[i]: val_class_recall_scores[i] for i in range(len(label_id_to_label_text))}
    val_class_precision_scores_dict = {label_id_to_label_text[i]: val_class_precision_scores[i] for i in range(len(label_id_to_label_text))}

    logger.log_metrics(metrics={'avg_train_loss': avg_train_loss[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_loss': avg_val_loss[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_f1_score': avg_val_f1_score[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_precision_score': avg_val_precision_score[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_recall_score': avg_val_recall_score[epoch]}, step=epoch)
    logger.log_metrics(metrics={'avg_val_accuracy_score': avg_val_accuracy_score[epoch]}, step=epoch)
    logger.experiment.add_scalars('val_class_f1_scores', val_class_f1_scores_dict, global_step=epoch)
    logger.experiment.add_scalars('val_class_recall_scores', val_class_recall_scores_dict, global_step=epoch)
    logger.experiment.add_scalars('val_class_precision_scores', val_class_precision_scores_dict, global_step=epoch)

logger.log_metrics(metrics={'avg_test_loss': result['eval_loss']}, step=0)
logger.log_metrics(metrics={'avg_test_f1_score': avg_test_f1_score}, step=0)
logger.log_metrics(metrics={'avg_test_precision_score': avg_test_precision_score}, step=0)
logger.log_metrics(metrics={'avg_test_recall_score': avg_test_recall_score}, step=0)
logger.log_metrics(metrics={'avg_test_accuracy_score': avg_test_accuracy_score}, step=0)
logger.experiment.add_scalars('test_class_f1_scores', test_class_f1_scores_dict, global_step=0)
logger.experiment.add_scalars('test_class_recall_scores', test_class_recall_scores_dict, global_step=0)
logger.experiment.add_scalars('test_class_precision_scores', test_class_precision_scores_dict, global_step=0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Use your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# predictions, raw_outputs = model.predict(['This thing is entirely different from the other thing. '])
