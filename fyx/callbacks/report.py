import random
import shutil

import pandas as pd

from fyx.callbacks.callback import Callback
from fyx.utils import to_numpy

class Report(Callback):
    def __init__(self, experiment_path, report_fn, log_to_track='val_total_loss', mode='min'):
        self.report_fn = report_fn
        self.experiment_path = experiment_path
        self.log_to_track = log_to_track
        self.mode = 'min'
        self.value = float('inf') if mode == 'min' else 0.0

    def on_epoch_begin(self, _):
        self.train_records = []
        self.validation_records = []

    def on_train_batch_end(self, _, outputs, batch):
        self.train_records.extend(self.report_fn(outputs, batch))

    def on_validation_batch_end(self, logs, outputs, batch):
        self.validation_records.extend(self.report_fn(outputs, batch))

    def on_epoch_end(self, logs):
        value = logs[self.log_to_track]
        if self.mode == 'min':
            update_needed = self.value > value
        else:
            update_needed = self.value < value

        train_df = pd.DataFrame(random.sample(self.train_records, min(len(self.train_records), 15000)))
        validation_df = pd.DataFrame(random.sample(self.validation_records, min(len(self.validation_records), 15000)))

        train_df.to_hdf(f'{self.experiment_path}/last-report-train.hdf', key='index')
        validation_df.to_hdf(f'{self.experiment_path}/last-report-valid.hdf', key='index')

        if update_needed:
            shutil.copyfile(f'{self.experiment_path}/last-report-train.hdf', f'{self.experiment_path}/best-report-train.hdf')
            shutil.copyfile(f'{self.experiment_path}/last-report-valid.hdf', f'{self.experiment_path}/best-report-valid.hdf')
            self.value = value
