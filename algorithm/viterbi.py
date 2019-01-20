from algorithm.algorithm import Algorithm
from experiment.experiment import Experiment
import numpy as np


class Viterbi(Algorithm):
    """Viterbi algorithm class (based on hmmlearn's Viterbi algorithm)"""
    def __init__(self, name):
        super().__init__(name)

    def run(self):
        """running the algorithm for each sample"""
        self.y_pred = np.zeros(self.experiment.y_test.shape, dtype=int)
        for i in range(0, self.experiment.y_test.shape[0]):
            self.y_pred[i, :] = self.decode(self.experiment.X_test[i,:].reshape(-1,1))[1]

    def decode(self, X):
        """sequence estimating using the Viterbi algorithm implementation of hmmlearn"""
        return self.experiment.hmm.decode(X, algorithm="viterbi")


if __name__ == "__main__":
    experiment = Experiment(trans_file="test_hmm_transition_matrix.csv",
                            emis_file="hmm_observations_emission.csv",
                            n_train_samples=2, n_test_samples=2, sample_length=100, seed=1)
    experiment.initialize()
    viterbi = Viterbi(experiment)
    print(experiment.y_test)
    viterbi.run()
    print(viterbi.y_pred)

