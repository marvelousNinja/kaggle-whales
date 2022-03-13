from fyx.callbacks.callback import Callback

class Dump(Callback):
    def __init__(self, value_getter, file_path):
        self.value_getter = value_getter
        self.file_path = file_path

    def on_epoch_end(self, logs):
        with open(self.file_path, 'w') as f:
            f.write(self.value_getter(logs))