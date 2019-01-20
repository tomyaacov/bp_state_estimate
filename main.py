from algorithm.genetic_filter import GeneticFilter
from algorithm.particle_filter import ParticleFilter
from algorithm.viterbi import Viterbi
from experiment.experiment import Experiment
import numpy as np
from itertools import product
import os
from experiment.result_analysis import ResultAnalysis

"""Main file: running the experiments with predetermined settings"""

if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
    # listing the possible separable states given the hmm
    possible_states = [list(x) for x in list(product(np.arange(4), np.arange(4)))]
    # listing the algorithms investigated in the experiment
    gfilter4 = GeneticFilter("AGF4", possible_states, genetic_operation_resolution=4)
    gfilter5 = GeneticFilter("AGF5", possible_states, genetic_operation_resolution=5)
    gfilter6 = GeneticFilter("AGF6", possible_states, genetic_operation_resolution=6)
    pfilter = ParticleFilter("APF")
    viterbi = Viterbi("VA")
    algorithms = [gfilter4, gfilter5, gfilter6, pfilter, viterbi]
    # creating the experiment
    experiment = Experiment(algorithms=algorithms,
                            trans_file=os.path.join(DATA_DIR, 'test_2_hmm_transition_matrix.csv'),
                            emis_file=os.path.join(DATA_DIR, 'hmm_observations_emission.csv'),
                            n_train_samples=10, n_test_samples=5000, sample_length=100, seed=1)
    # initialize and run the experiment
    experiment.initialize()
    experiment.run()
    # analyzing the results obtained in the experiment
    result = ResultAnalysis(experiment, algorithms)
    result.run()
