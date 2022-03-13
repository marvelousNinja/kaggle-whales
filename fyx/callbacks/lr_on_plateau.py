from fyx.callbacks.callback import Callback

class LROnPlateau(Callback):
    def __init__(self, log_to_track, optimizer, mode, factor, patience, min_lr, logger):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.plateau_counter = 0
        self.log_to_track = log_to_track
        self.best_value = float('inf') if mode == 'min' else 0.0
        self.min_lr = min_lr
        self.logger = logger
        self.threshold = 1e-4

    def on_epoch_end(self, logs):
        logs['train_lr'] = self.optimizer.param_groups[0]['lr']

        current_value = logs[self.log_to_track]
        if self.mode == 'min':
            value_improved = current_value < self.best_value * (1 - self.threshold)
        else:
            value_improved = current_value > self.best_value * (1 + self.threshold)

        if value_improved:
            self.plateau_counter = 0
            self.best_value = current_value
            return

        self.plateau_counter += 1
        if self.plateau_counter < self.patience: return
        self.plateau_counter = 0
        for param_group in self.optimizer.param_groups:
            new_lr = max(self.min_lr, param_group['lr'] * self.factor)
            param_group['lr'] = new_lr
            self.logger(f'LR on Plateau: LR reduced to {new_lr:.8f}')
