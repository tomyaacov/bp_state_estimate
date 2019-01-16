from abc import abstractmethod


class Algorithm:

    def __init__(self, experiment):
        self.experiment = experiment
        self.y_pred = None

    @abstractmethod
    def run(self):
        pass

