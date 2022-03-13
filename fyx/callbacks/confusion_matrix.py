import numpy as np
import torch
from tabulate import tabulate

from fyx.callbacks.callback import Callback
from fyx.utils import to_numpy

def confusion_matrix(pred_labels, true_labels, labels):
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    columns = [list(map(lambda label: f'Pred {label}', labels))]
    for true_label in labels:
        counts = []
        preds_for_label = pred_labels[np.argwhere(true_labels == true_label)]
        for predicted_label in labels:
            counts.append((preds_for_label == predicted_label).sum())
        columns.append(counts)

    headers = list(map(lambda label: f'True {label}', labels))
    rows = np.column_stack(columns)
    return tabulate(rows, headers, 'grid')

class ConfusionMatrix(Callback):
    def __init__(self, labels, logger):
        self.logger = logger
        self.labels = labels
        self.preds = []
        self.gt = []

    def on_epoch_begin(self, logs):
        self.preds = []
        self.gt = []

    def on_validation_batch_end(self, _, outputs, batch):
        self.preds.extend(to_numpy(torch.argmax(outputs['target'], dim=1).long()).reshape(-1))
        self.gt.extend(to_numpy(batch['target']).reshape(-1))

    def on_epoch_end(self, _):
        self.logger(confusion_matrix(self.preds, self.gt, self.labels))
