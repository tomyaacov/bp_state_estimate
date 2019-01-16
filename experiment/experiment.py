from hmm_model.hmm import build_hmm
from numpy import genfromtxt
import numpy as np
from datetime import datetime


class Experiment:

    def __init__(self, algorithms, trans_file, emis_file, n_train_samples=20, n_test_samples=20, sample_length=2000, seed=1):
        self.algorithms = algorithms
        self.trans_file = trans_file
        self.emis_file = emis_file
        self.seed = seed
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.sample_length = sample_length
        self.hmm = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def initialize(self):
        trans = genfromtxt(self.trans_file, delimiter=',')
        emis = genfromtxt(self.emis_file, delimiter=',')
        self.hmm = build_hmm(trans, emis, self.seed)
        self.generate_samples()

    def generate_samples(self):
        self.X_train = np.zeros([self.n_train_samples, self.sample_length]).astype(int)
        self.y_train = np.zeros([self.n_train_samples, self.sample_length]).astype(int)
        for i in range(self.n_train_samples):
            current_sample = self.hmm.sample(self.sample_length, random_state=self.hmm.random_state+i)
            self.X_train[i, :] = current_sample[0].ravel()
            self.y_train[i, :] = current_sample[1].ravel()

        self.X_test = np.zeros([self.n_test_samples, self.sample_length]).astype(int)
        self.y_test = np.zeros([self.n_test_samples, self.sample_length]).astype(int)
        for i in range(self.n_test_samples):
            current_sample = self.hmm.sample(self.sample_length, random_state=self.hmm.random_state+self.n_train_samples+i)
            self.X_test[i, :] = current_sample[0].ravel()
            self.y_test[i, :] = current_sample[1].ravel()

    def run(self):
        for algorithm in self.algorithms:
            algorithm.experiment = self
            start = datetime.now()
            algorithm.run()
            algorithm.run_time = round((datetime.now()-start).total_seconds(), 2)

if __name__ == "__main__":
    experiment = Experiment(trans_file="hmm_model/hmm_transition_matrix.csv",
                            emis_file="hmm_model/hmm_observations_emission.csv",
                            n_train_samples=2, n_test_samples=2, sample_length=20, seed=1)
    experiment.initialize()
    print(experiment.X_train)
    print(experiment.X_test)
    print(experiment.y_train)
    print(experiment.y_test)


