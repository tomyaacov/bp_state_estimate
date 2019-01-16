import os
import sys
import datetime
from algorithm.genetic_filter import GeneticFilter
from algorithm.viterbi import Viterbi
import numpy as np
from itertools import product
from experiment.experiment import Experiment
import matplotlib.pyplot as plt
import pandas as pd


class ResultAnalysis:

    def __init__(self, experiment, algorithms):
        self.experiment = experiment
        self.algorithms = algorithms

    def run(self):
        self.RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(sys.modules['__main__'].__file__), 'results', self.modify_date_time()))
        os.mkdir(self.RESULTS_DIR)
        self.compute_mean_accuracy()
        self.mean_accuracy_plot()
        self.compute_mean_adjusted_accuracy()
        self.mean_adjusted_accuracy_plot()
        self.create_summary_table()
        self.mean_population_fitness_plot()

    def create_summary_table(self):
        self.summary_table = pd.DataFrame(columns=['algorithm', 'mean accuracy', 'mean adjusted accuracy', 'run time'])
        for i in range(len(self.algorithms)):
            algorithm_name = self.algorithms[i].name
            self.summary_table.loc[i] = [algorithm_name,
                                         np.mean(self.mean_accuracy[algorithm_name]),  # TODO: not sure its the best way to compute mean
                                         np.mean(self.mean_adjusted_accuracy[algorithm_name]),
                                         self.algorithms[i].run_time]
        self.summary_table.to_csv(os.path.join(self.RESULTS_DIR, 'summary_table.csv'), index=False)

    def compute_mean_accuracy(self):
        self.mean_accuracy = {}
        for algirithm in self.algorithms:
            accuracy = algirithm.y_pred == self.experiment.y_test
            self.mean_accuracy[algirithm.name] = np.mean(accuracy, axis=0)

    def mean_accuracy_plot(self):
        plt.figure()
        for key in self.mean_accuracy.keys():
            plt.plot(self.mean_accuracy[key], label=key)
        plt.ylabel('mean accuracy')
        plt.xlabel('time')
        plt.title('mean accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.RESULTS_DIR, 'mean_accuracy_plot.png'))

    def compute_mean_adjusted_accuracy(self, base_state=0):
        self.mean_adjusted_accuracy = {}
        for algorithm in self.algorithms:
            accuracy = (algorithm.y_pred == base_state) == (self.experiment.y_test == base_state)
            self.mean_adjusted_accuracy[algorithm.name] = np.mean(accuracy, axis=0)

    def mean_adjusted_accuracy_plot(self):
        plt.figure()
        for key in self.mean_adjusted_accuracy.keys():
            plt.plot(self.mean_adjusted_accuracy[key], label=key)
        plt.ylabel('mean adjusted accuracy')
        plt.xlabel('time')
        plt.title('mean adjusted accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.RESULTS_DIR, 'mean_adjusted_accuracy_plot.png'))

    def mean_population_fitness_plot(self):
        plt.figure()
        for algorithm in self.algorithms:
            if isinstance(algorithm, GeneticFilter):
                plt.plot(algorithm.get_mean_fitness(), label=algorithm.name)
        plt.ylabel('mean population fitness')
        plt.xlabel('time')
        plt.title('mean population fitness')
        plt.legend()
        plt.savefig(os.path.join(self.RESULTS_DIR, 'mean_population_fitness_plot.png'))

    @staticmethod
    def modify_date_time():
        result = str(datetime.datetime.today().replace(microsecond=0))
        for char in "-: ":
            result = result.replace(char, "_")
        return result


if __name__=="__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
    experiment = Experiment(trans_file=os.path.join(DATA_DIR, 'test_hmm_transition_matrix.csv'),
                            emis_file=os.path.join(DATA_DIR, 'hmm_observations_emission.csv'),
                            n_train_samples=2, n_test_samples=2, sample_length=20, seed=1)
    experiment.initialize()
    possible_states = [list(x) for x in list(product(np.arange(4), np.arange(4)))]
    filter = GeneticFilter(experiment, "a", possible_states)
    filter.run()
    viterbi = Viterbi(experiment, "v")
    viterbi.run()
    result = ResultAnalysis(experiment, [filter, viterbi])
    result.run()

