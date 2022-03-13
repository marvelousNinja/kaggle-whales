import matplotlib.pyplot as plt

from airbus.callbacks.callback import Callback

class LRRangeTest(Callback):
    def __init__(self, min_lr, max_lr, num_iter, optimizer, image_logger):
        self.image_logger = image_logger
        self.lrs = []
        self.losses = []
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_lr = min_lr
        self.iter_counter = 0
        self.num_iter = num_iter
        self.optimizer = optimizer

    def on_epoch_begin(self, _):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def on_train_batch_end(self, logs, *_):
        self.iter_counter += 1
        self.lrs.append(self.current_lr)
        self.losses.append(logs['batch_loss'])

        self.current_lr = self.min_lr + (self.max_lr - self.min_lr) * self.iter_counter / self.num_iter
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def on_epoch_end(self, _):
        plt.figure()
        plt.plot(self.lrs, self.losses, label='LR for train batch')
        plt.legend()
        self.image_logger(plt.gcf())

