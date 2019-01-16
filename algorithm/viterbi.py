from algorithm.algorithm import Algorithm
from experiment import Experiment
import numpy as np


class Viterbi(Algorithm):
    def __init__(self, experiment):
        super().__init__(experiment)

    def run(self):
        self.y_pred = np.zeros(self.experiment.y_test.shape)
        for i in range(0, self.experiment.y_test.shape[0]):
            self.y_pred[i, :] = self.decode(self.experiment.X_test[i,:].reshape(-1,1))[1]

    def decode(self, X):
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

