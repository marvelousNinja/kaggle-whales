import matplotlib.pyplot as plt

from airbus.callbacks.callback import Callback
from airbus.utils import to_numpy

class Histogram(Callback):
    def __init__(self, image_logger, metric_fn):
        self.image_logger = image_logger
        self.metric_fn = metric_fn
        self.values = None

    def on_epoch_begin(self, _):
        self.values = []

    def on_validation_batch_end(self, logs, outputs, batch):
        self.values.extend(to_numpy(self.metric_fn(outputs, batch, average=False)))

    def on_epoch_end(self, logs):
        plt.hist(self.values, bins=20)
        plt.title(self.metric_fn.__name__)
        self.image_logger(plt.gcf())
