import torch
from pytorch_lightning.metrics.classification import Accuracy, Precision, Recall, Fbeta
from pytorch_lightning.metrics.utils import METRIC_EPS
from config import configuration as cfg


def accuracy_score(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    accuracy = Accuracy(threshold=cfg['training']['threshold'])
    return accuracy(predictions, labels)


def precision_score(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    precision = Precision(num_classes=len(labels[0]), average='micro', multilabel=True, threshold=cfg['training']['threshold'])
    return precision(predictions, labels)


def class_wise_precision_scores(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    precision = Precision(num_classes=len(labels[0]), average='macro', multilabel=True, threshold=cfg['training']['threshold'])
    precision(predictions, labels)
    class_wise_scores = precision.true_positives.float() / (precision.predicted_positives + METRIC_EPS)
    return class_wise_scores


def recall_score(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    recall = Recall(num_classes=len(labels[0]), average='micro', multilabel=True, threshold=cfg['training']['threshold'])
    return recall(predictions, labels)


def class_wise_recall_scores(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    recall = Recall(num_classes=len(labels[0]), average='macro', multilabel=True, threshold=cfg['training']['threshold'])
    recall(predictions, labels)
    class_wise_scores = recall.true_positives.float() / (recall.actual_positives + METRIC_EPS)
    return class_wise_scores


def f1_score(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    f_beta = Fbeta(num_classes=len(labels[0]), average='micro', multilabel=True, threshold=cfg['training']['threshold'])
    return f_beta(predictions, labels)


def class_wise_f1_scores(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    f_beta = Fbeta(num_classes=len(labels[0]), average='macro', multilabel=True, threshold=cfg['training']['threshold'])
    f_beta(predictions, labels)
    precision = f_beta.true_positives.float() / (f_beta.predicted_positives + METRIC_EPS)
    recall = f_beta.true_positives.float() / (f_beta.actual_positives + METRIC_EPS)
    class_wise_scores = (precision * recall) / (precision + recall + METRIC_EPS)
    return class_wise_scores


if __name__ == "__main__":

    truth = torch.Tensor([[1, 1, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 0, 0, 1]])
    pred = torch.Tensor([[0.2, 0.7, 0.6, 0.8, 0.7], [1, 0.2, 0.3, 0.76, 0.98], [0.56, 0.67, 0.78, 0.23, 0.87]])

    print(accuracy_score(pred, truth))
    print(precision_score(pred, truth))
    print(recall_score(pred, truth))
    print(class_wise_precision_scores(pred, truth))
    print(class_wise_recall_scores(pred, truth))
    print(class_wise_f1_scores(pred, truth))
