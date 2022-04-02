from torch.utils.tensorboard import SummaryWriter

from fyx.callbacks.callback import Callback


class TensorboardMonitor(Callback):
    def __init__(self, experiment_dir, visualize_fn=lambda *args: None, end_of_epoch_fn=lambda *args: None):
        self.writer = SummaryWriter(f'{experiment_dir}/tensorboard', flush_secs=10)
        self.epoch_counter = 0
        self.batch_counter = 0
        self.visualize_fn = visualize_fn
        self.end_of_epoch_fn = end_of_epoch_fn
        self.experiment_dir = experiment_dir

    def on_train_batch_end(self, logs, outputs, batch):
        return
        if (self.batch_counter % 100 == 0) and self.batch_counter > 0:
            self.batch_counter += 1
            self.visualize_fn(self.writer, 'train', outputs, batch, self.batch_counter)
        else:
            self.batch_counter += 1

    def on_validation_batch_end(self, logs, outputs, batch):
        # TODO AS: For now, log every batch in validation for timelapse logs
        # This assumes that images in validation are unique
        # TODO AS: Optionally disable? Slows the training down..
        # self.visualize_fn(self.writer, 'val', outputs, batch, self.epoch_counter)
        pass

    def on_epoch_end(self, logs):
        for key, value in logs.items():
            key = key.replace('_', '/')
            self.writer.add_scalar(key, value, self.epoch_counter)

        self.end_of_epoch_fn(self.writer, self.epoch_counter, self.experiment_dir)
        self.epoch_counter += 1