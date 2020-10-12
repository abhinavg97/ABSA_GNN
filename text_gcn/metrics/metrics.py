import torch


def class_wise_precision_scores(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    pass


def class_wise_recall_scores(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    pass


def class_wise_f1_scores(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    pass


def f1_scores_average(predictions, labels):
    predictions = torch.Tensor(predictions)
    labels = torch.Tensor(labels)
    pass


if __name__ == "__main__":

    truth = torch.Tensor([[1, 1, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 0, 0, 1]])
    pred = torch.Tensor([[0, 1, 1, 1, 1], [1, 0, 0, 1, 1], [1, 1, 1, 0, 1]])
