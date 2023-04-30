import torchmetrics
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy(preds, target):
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)
    return accuracy(preds, target)*100

def precision(preds, target):
    precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=100).to(device)
    return precision(preds, target)*100

def recall(preds, target):
    recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=100).to(device)
    return recall(preds, target)*100

def f1score(preds, target):
    f1score = torchmetrics.F1Score(task="multiclass", num_classes=100).to(device)
    return f1score(preds, target)*100

