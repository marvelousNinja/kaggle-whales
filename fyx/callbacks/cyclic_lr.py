import torch.optim.lr_scheduler

from fyx.callbacks.callback import Callback

class CyclicLR(Callback):
    def __init__(self, **kwargs):
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(**kwargs)

    def on_train_batch_end(self, *_):
        self.scheduler.step()
