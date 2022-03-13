import matplotlib.pyplot as plt

from fyx.callbacks.callback import Callback

class LearningCurve(Callback):
    def __init__(self, logs_to_track, image_logger):
        self.image_logger = image_logger
        self.history = {}
        self.logs_to_track = logs_to_track
        for name in logs_to_track: self.history[name] = []

    def on_epoch_end(self, logs):
        for name in self.logs_to_track:
            self.history[name].append(logs[name])
            plt.plot(self.history[name], label=name)
        plt.legend()
        self.image_logger(plt.gcf())
