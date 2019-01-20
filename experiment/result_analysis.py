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
    """Class ResultAnalysis which analyze the results obtained in experiment"""
    def __init__(self, experiment, algorithms):
        self.experiment = experiment
        self.algorithms = algorithms

    def run(self):
        """Running the analysis given the experiment results and the """
        self.RESULTS_DIR = os.path.abspath(
            os.path.join(os.path.dirname(sys.modules['__main__'].__file__), 'results', self.modify_date_time()))
        os.mkdir(self.RESULTS_DIR)
        self.compute_mean_accuracy()
        self.mean_accuracy_plot()
        self.compute_mean_binary_accuracy()
        self.mean_binary_accuracy_plot()
        self.compute_mean_bthread_binary_accuracy()
        self.mean_adjusted_bthread_binary_plot()
        self.create_summary_table()
        self.mean_population_fitness_plot()

    def create_summary_table(self):
        """Create the summary table with the evaluation metrices for each algorithm being examined"""
        self.summary_table = pd.DataFrame(columns=['Algorithm', 'Mean Acc.', 'Mean Binary Acc.', 'Mean B-thread Binary Acc.'])
        for i in range(len(self.algorithms)):
            algorithm_name = self.algorithms[i].name
            self.summary_table.loc[i] = [algorithm_name,
                                         round(np.mean(self.mean_accuracy[algorithm_name]), 3),
                                         round(np.mean(self.mean_binary_accuracy[algorithm_name]), 3),
                                         round(np.mean(self.mean_bthread_binary_accuracy[algorithm_name]), 3)]
        self.summary_table.to_csv(os.path.join(self.RESULTS_DIR, 'summary_table.csv'), index=False)

    def compute_mean_accuracy(self):
        """Compute the mean accuracy for each algorithm"""
        self.mean_accuracy = {}
        for algorithm in self.algorithms:
            accuracy = algorithm.y_pred == self.experiment.y_test
            self.mean_accuracy[algorithm.name] = np.mean(accuracy, axis=0)

    def mean_accuracy_plot(self):
        """Plotting the mean accuracy for each algorithm"""
        plt.figure()
        for key in self.mean_accuracy.keys():
            plt.plot(self.mean_accuracy[key], label=key)
        plt.ylabel('mean accuracy')
        plt.xlabel('time')
        plt.legend()
        plt.axis([0, self.experiment.sample_length, 0, 1])
        plt.grid(True)
        plt.savefig(os.path.join(self.RESULTS_DIR, 'mean_accuracy_plot.png'))

    def compute_mean_binary_accuracy(self, base_state=0):  # need to be adjusted to different hmms
        """Compute the mean binary accuracy for each algorithm"""
        self.mean_binary_accuracy = {}
        for algorithm in self.algorithms:
            accuracy = (algorithm.y_pred == base_state) == (self.experiment.y_test == base_state)
            self.mean_binary_accuracy[algorithm.name] = np.mean(accuracy, axis=0)

    def mean_binary_accuracy_plot(self):
        """Plotting the mean binary accuracy for each algorithm"""
        plt.figure()
        for key in self.mean_binary_accuracy.keys():
            plt.plot(self.mean_binary_accuracy[key], label=key)
        plt.ylabel('mean binary accuracy')
        plt.xlabel('time')
        plt.legend()
        plt.axis([0, self.experiment.sample_length, 0, 1])
        plt.grid(True)
        plt.savefig(os.path.join(self.RESULTS_DIR, 'mean_binary_accuracy_plot.png'))

    def compute_mean_bthread_binary_accuracy(self, threads_groups=[[0], [4, 8, 12], [1, 2, 3], [5, 6, 7, 9, 10, 11, 13, 14, 15]]):  # need to be adjusted to different hmms
        """Compute the mean bthread binary accuracy for each algorithm"""
        self.mean_bthread_binary_accuracy = {}
        real_value = self.experiment.y_test
        for i in range(len(threads_groups)):
            real_value[np.isin(real_value, threads_groups[i])] = i
        for algorithm in self.algorithms:
            algorithm_prediction = algorithm.y_pred
            for i in range(len(threads_groups)):
                algorithm_prediction[np.isin(algorithm_prediction, threads_groups[i])] = i
            accuracy = algorithm_prediction == real_value
            self.mean_bthread_binary_accuracy[algorithm.name] = np.mean(accuracy, axis=0)

    def mean_adjusted_bthread_binary_plot(self):
        """Plotting the mean bthread binary accuracy for each algorithm"""
        plt.figure()
        for key in self.mean_bthread_binary_accuracy.keys():
            plt.plot(self.mean_bthread_binary_accuracy[key], label=key)
        plt.ylabel('mean b-thread binary accuracy')
        plt.xlabel('time')
        plt.legend()
        plt.axis([0, self.experiment.sample_length, 0, 1])
        plt.grid(True)
        plt.savefig(os.path.join(self.RESULTS_DIR, 'mean_bthread_binary_accuracy_plot.png'))

    def mean_population_fitness_plot(self):
        """Plotting the mean fitness for each genetic algorithm"""
        plt.figure()
        for algorithm in self.algorithms:
            if isinstance(algorithm, GeneticFilter):
                x, y = algorithm.get_mean_fitness()
                plt.plot(x, y, label=algorithm.name)
        plt.ylabel('mean population fitness')
        plt.xlabel('time')
        plt.legend()
        plt.axis([0, self.experiment.sample_length, 0, 0.05])
        plt.grid(True)
        plt.savefig(os.path.join(self.RESULTS_DIR, 'mean_population_fitness_plot.png'))

    @staticmethod
    def modify_date_time():
        """Utility method for experiment results folder saving"""
        result = str(datetime.datetime.today().replace(microsecond=0))
        for char in "-: ":
            result = result.replace(char, "_")
        return result


if __name__ == "__main__":
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
