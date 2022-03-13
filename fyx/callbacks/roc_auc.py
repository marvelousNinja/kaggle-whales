import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from fyx.callbacks.callback import Callback
from fyx.utils import to_numpy

class RocAuc(Callback):
    def __init__(self):
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []

    def on_epoch_begin(self, logs):
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []

        logs['train_roc_auc'] = 0
        logs['val_roc_auc'] = 0

    def on_train_batch_end(self, logs, outputs, batch):
        self.train_preds.extend(to_numpy(torch.sigmoid(outputs['classification_logits'])[:, 0]))
        self.train_labels.extend(to_numpy(batch['transaction_performed']).astype(np.uint8))

    def on_validation_batch_end(self, logs, outputs, batch):
        self.val_preds.extend(to_numpy(torch.sigmoid(outputs['classification_logits'])[:, 0]))
        self.val_labels.extend(to_numpy(batch['transaction_performed']).astype(np.uint8))

    def on_epoch_end(self, logs):
        logs['train_roc_auc'] = roc_auc_score(self.train_labels, self.train_preds)
        logs['val_roc_auc'] = roc_auc_score(self.val_labels, self.val_preds)
