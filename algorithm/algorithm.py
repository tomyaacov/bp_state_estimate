from abc import abstractmethod


class Algorithm:
    """Abstract class Algorithm"""
    def __init__(self, name, experiment=None):
        self.experiment = experiment
        self.name = name
        self.y_pred = None
        self.run_time = None

    @abstractmethod
    def run(self):
        pass

