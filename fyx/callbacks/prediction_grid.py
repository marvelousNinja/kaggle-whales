import matplotlib.pyplot as plt
import numpy as np

from airbus.callbacks.callback import Callback
from airbus.utils import to_numpy

def visualize_predictions(image_logger, max_samples, metrics, predictions_and_gt):
    num_samples = min(len(metrics), max_samples)
    order = np.argsort(metrics)

    # TODO AS: Clean this mess up
    predictions_and_gt_ = []
    metrics_ = []
    for i in order:
        predictions_and_gt_.append(predictions_and_gt[i])
        metrics_.append(metrics[i])
    predictions_and_gt = predictions_and_gt_
    predictions_and_gt = predictions_and_gt[:num_samples]
    metrics = metrics_
    metrics = metrics[:num_samples]
    # TODO AS: End of the mess

    samples_per_row = 16
    num_rows = int(np.ceil(num_samples / samples_per_row)) * 2
    plt.figure(figsize=(6, 1 * num_rows))

    for i in range(num_samples):
        plt.subplot(num_rows, samples_per_row, (i // samples_per_row) * samples_per_row + i + 1)
        plt.title(f'{metrics[i]:.1f}')
        plt.imshow(predictions_and_gt[i][0], vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(num_rows, samples_per_row, (i // samples_per_row + 1) * samples_per_row + i + 1)
        plt.imshow(predictions_and_gt[i][1])
        plt.xticks([])
        plt.yticks([])
    plt.gcf().tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    image_logger(plt.gcf())

class PredictionGrid(Callback):
    def __init__(self, max_samples, image_logger, metric_fn, pred_fn):
        self.max_samples = max_samples
        self.image_logger = image_logger
        self.metric_fn = metric_fn
        self.pred_fn = pred_fn
        self.metrics = []
        self.predictions_and_gt = []

    def on_epoch_begin(self, _):
        self.metrics = []
        self.predictions_and_gt = []

    def on_validation_batch_end(self, logs, outputs, batch):
        self.metrics.extend(to_numpy(self.metric_fn(outputs, batch, average=False)))
        preds = to_numpy(self.pred_fn(outputs, batch))
        for i in range(len(preds[0])): self.predictions_and_gt.append((preds[0][i], preds[1][i]))

    def on_epoch_end(self, _):
        visualize_predictions(self.image_logger, self.max_samples, self.metrics, self.predictions_and_gt)
