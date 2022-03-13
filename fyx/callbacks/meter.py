from collections import defaultdict

import numpy as np

from fyx.callbacks.callback import Callback

class Meter(Callback):
    def __init__(self, name, map_fn, reduce_fn=np.mean, reduce_once=False, only_val=False):
        self.accumulator = defaultdict(lambda: [])
        self.name = name
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn
        self.reduce_once = reduce_once
        self.only_val = only_val

    def on_epoch_begin(self, logs):
        self.accumulator = defaultdict(lambda: [])

    def on_train_batch_end(self, logs, outputs, batch):
        if self.only_val: return
        logs[f'train_{self.name}'] = 0 if self.reduce_once else self.reduce_fn([self.map_fn(outputs, batch)])
        self.accumulator[f'train_{self.name}'].append(self.map_fn(outputs, batch))

    def on_validation_batch_end(self, logs, outputs, batch):
        logs[f'val_{self.name}'] = 0 if self.reduce_once else self.reduce_fn([self.map_fn(outputs, batch)])
        self.accumulator[f'val_{self.name}'].append(self.map_fn(outputs, batch))

    def on_epoch_end(self, logs):
        for name, values in self.accumulator.items():
            out = self.reduce_fn(values)
            if isinstance(out, dict):
                for key, value in out.items():
                    logs[f'{name}_{key}'] = value
            else:
                logs[name] = out
