import math

import matplotlib.pyplot as plt

from airbus.callbacks.callback import Callback
from airbus.utils import to_numpy

class WeightGrid(Callback):
    def __init__(self, model, image_logger, max_samples):
        self.model = model
        self.image_logger = image_logger
        self.grads = {}
        self.batch_counter = 0
        self.max_samples = max_samples

    def on_train_batch_end(self, *_):
        state_dict = self.model.state_dict(keep_vars=True)
        for layer_name, weights in reversed(state_dict.items()):
            if weights is not None and weights.grad is not None:
                if self.grads.get(layer_name) is None:
                    self.grads[layer_name] = weights.grad.clone()
                else:
                    self.grads[layer_name] += weights.grad

                if len(self.grads) >= self.max_samples:
                    break
        self.batch_counter += 1

    def on_epoch_end(self, logs):
        state_dict = self.model.state_dict(keep_vars=True)
        samples_per_row = 6
        num_samples = len(self.grads)
        num_rows = math.ceil(num_samples / samples_per_row) * 2
        plt.figure(figsize=(16, 1.6 * num_rows))
        for i, (layer_name, grads) in enumerate(self.grads.items()):
            plt.subplot(num_rows, samples_per_row, (i // samples_per_row) * samples_per_row + i + 1)
            plt.title(layer_name)
            flat_weights = to_numpy(state_dict[layer_name]).reshape(-1)
            plt.hist(flat_weights, bins=50)

            plt.subplot(num_rows, samples_per_row, (i // samples_per_row + 1) * samples_per_row + i + 1)
            plt.title(layer_name)
            flat_grads = to_numpy(grads / self.batch_counter).reshape(-1)
            plt.hist(flat_grads, bins=50, color='green')
        plt.tight_layout()
        self.image_logger(plt.gcf())
        self.batch_counter = 0
        self.grads = {}
