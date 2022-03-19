import copy

import torch

from fyx.callbacks.callback import Callback

def save_checkpoint(model, path):
    model = copy.deepcopy(model).to('cpu')
    if not isinstance(model, torch.jit.ScriptModule):
        torch.save(model, path + '.serialized')
        # TODO AS: Tracing leads to memory leaks for some reason
        model = torch.jit.trace(model, model.sample_input())
    torch.jit.save(model, path)

def load_checkpoint(path):
    if path.endswith('.serialized'):
        return torch.load(path, map_location='cpu')
    else:
        return torch.jit.load(path, map_location='cpu')

class ModelCheckpoint(Callback):
    def __init__(self, model, experiment_dir, log_to_track, mode, logger=None):
        self.epoch = 0
        self.model = model
        self.logger = logger
        self.mode = mode
        self.value = float('inf') if mode == 'min' else 0.0
        self.log_to_track = log_to_track
        self.experiment_dir = experiment_dir

    def on_epoch_end(self, logs):
        value = logs[self.log_to_track]
        if self.mode == 'min':
            update_needed = self.value > value
        else:
            update_needed = self.value < value

        save_checkpoint(self.model, f'{self.experiment_dir}/last.pth')

        if update_needed:
            save_checkpoint(self.model, f'{self.experiment_dir}/best.pth')
            self.value = value
            if self.logger: self.logger(f'Checkpoint saved {self.experiment_dir}/best.pth')

        self.epoch += 1
