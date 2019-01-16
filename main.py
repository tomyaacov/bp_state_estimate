from algorithm.genetic_filter import GeneticFilter
from algorithm.viterbi import Viterbi
from experiment.experiment import Experiment
import numpy as np
from itertools import product
import os

if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
    experiment = Experiment(trans_file=os.path.join(DATA_DIR, 'test_hmm_transition_matrix.csv'),
                            emis_file=os.path.join(DATA_DIR, 'hmm_observations_emission.csv'),
                            n_train_samples=2, n_test_samples=2, sample_length=20, seed=1)
    experiment.initialize()
    possible_states = [list(x) for x in list(product(np.arange(4),np.arange(4)))]
    filter = GeneticFilter(experiment, "a", possible_states)
    print(experiment.y_test)
    filter.run()
    print(filter.y_pred)
    viterbi = Viterbi(experiment, "v")
    viterbi.run()
    print(viterbi.y_pred)