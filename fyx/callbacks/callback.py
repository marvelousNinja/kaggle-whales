class Callback:
    def on_epoch_begin(self, logs):
        pass

    def on_epoch_end(self, logs):
        pass

    def on_train_batch_end(self, logs, outputs, batch):
        pass

    def on_validation_batch_end(self, logs, outputs, batch):
        pass
