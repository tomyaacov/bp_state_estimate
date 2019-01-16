from abc import abstractmethod


class Algorithm:

    def __init__(self, experiment, name):
        self.experiment = experiment
        self.name = name
        self.y_pred = None

    @abstractmethod
    def run(self):
        pass

