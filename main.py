from algorithm.genetic_filter import GeneticFilter
from algorithm.particle_filter import ParticleFilter
from algorithm.viterbi import Viterbi
from experiment.experiment import Experiment
import numpy as np
from itertools import product
import os
from experiment.result_analysis import ResultAnalysis

if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
    possible_states = [list(x) for x in list(product(np.arange(4), np.arange(4)))]
    gfilter = GeneticFilter("Genetic Filter", possible_states)
    viterbi = Viterbi("Viterbi")
    pfilter = ParticleFilter("Particle Filter")
    algorithms = [gfilter, viterbi, pfilter]
    experiment = Experiment(algorithms=algorithms,
                            trans_file=os.path.join(DATA_DIR, 'test_hmm_transition_matrix.csv'),
                            emis_file=os.path.join(DATA_DIR, 'hmm_observations_emission.csv'),
                            n_train_samples=10, n_test_samples=1000, sample_length=100, seed=1)
    experiment.initialize()
    experiment.run()
    print(experiment.y_test)
    print(gfilter.y_pred)
    print(viterbi.y_pred)
    print(pfilter.y_pred)
    result = ResultAnalysis(experiment, algorithms)
    result.run()