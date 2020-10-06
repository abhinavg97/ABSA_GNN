import torch
from pytorch_lightning.metrics.sklearns import F1, Precision, Recall


def class_wise_precision_scores(predictions, labels):

    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    metric = Precision(average='macro')
    class_precision_scores_list = metric(predictions, labels)

    return class_precision_scores_list


def class_wise_recall_scores(predictions, labels):

    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    metric = Recall(average='macro')
    class_recall_scores_list = metric(predictions, labels)

    return class_recall_scores_list


def class_wise_f1_scores(predictions, labels):

    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    metric = F1(average='macro')
    class_f1_scores_list = torch.Tensor([0 for i in range(len(predictions[0]))])
    for i, pred in enumerate(predictions):
        label_list = labels[i]
        for j, class_pred in enumerate(pred):
            class_f1_scores_list[j] += metric(class_pred.item() > 0.5, label_list[j].item()).item()
    class_f1_scores_list /= len(predictions)
    return class_f1_scores_list


def f1_scores_average(predictions, labels):

    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    metric = F1(average='macro')
    avg_f1_score = sum(list(map(lambda pred, y: metric(pred > 0.5, y), predictions, labels)))/predictions.shape[0]

    return avg_f1_score
