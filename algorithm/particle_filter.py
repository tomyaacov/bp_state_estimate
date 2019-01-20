from algorithm.algorithm import Algorithm
from experiment.experiment import Experiment
from hmm_model.hmm import *
from itertools import product


class ParticleFilter(Algorithm):
    """Class Particle Filter algorithm for the suggested adapted particle filter"""
    def __init__(self, name, population_size=5, num_of_sub_processes=2, weight_calculation_resolution=3):
        super().__init__(name)
        self.population_size = population_size
        self.num_of_sub_processes = num_of_sub_processes
        self.weight_calculation_resolution = weight_calculation_resolution

    def run(self):
        """running the algorithm for each sample"""
        self.y_pred = np.zeros(self.experiment.y_test.shape, dtype=int)
        for i in range(self.experiment.y_test.shape[0]):
            population = {0: self.initialize_population()}
            for j in range(self.experiment.y_test.shape[1]):
                # calculating fitness
                fitness = self.cal_pop_fitness_over_time(self.experiment.hmm,
                                                         [population[x] for x in
                                                          range(max(0, j - self.weight_calculation_resolution + 1), j + 1)],
                                                         [self.experiment.X_test[i, x] for x in
                                                          range(max(0, j - self.weight_calculation_resolution + 1), j + 1)])
                # state estimation
                self.y_pred[i, j] = many_to_one(population[j][np.argmax(fitness), :])
                # weighted sampling sampling
                sampled_population = self.roulette_wheel_selection(population[j], fitness)
                # transition sampling
                population[j + 1] = self.transition_population(sampled_population)

    def transition_population(self, population):
        """Propogate the particles given the transition matrix of the hmm"""
        next_population = np.zeros(population.shape, dtype=int)
        for i in range(population.shape[0]):
            next_population[i, :] = self.transition_sample(population[i, :])
        return next_population

    def transition_sample(self, individual):
        """sample a state from the transition matrix for particles propogation"""
        return one_to_many(np.random.choice(self.experiment.hmm.transmat_.shape[1],
                                            p=self.experiment.hmm.transmat_[many_to_one(individual), :]))

    def initialize_population(self):
        """Initialize population from the hmm's start state probability"""
        pop_size = (self.population_size, self.num_of_sub_processes)
        new_population = [
            one_to_many(np.random.choice(len(self.experiment.hmm.startprob_), p=self.experiment.hmm.startprob_)) for _
            in range(self.population_size)]
        return np.reshape(new_population, pop_size)

    def cal_pop_fitness(self, hmm, pop, observation):
        """calculate the population's fitness given the observations"""
        fitness = np.array([hmm.emissionprob_[many_to_one(x), observation] for x in pop])
        return fitness

    def cal_pop_fitness_over_time(self, hmm, pop_list, observation_list):
        """calculate the population's fitness given the observations over time"""
        fitness = np.ones(pop_list[0].shape[0], dtype=float)
        for i in range(len(pop_list)):
            fitness *= self.cal_pop_fitness(hmm, pop_list[i], observation_list[i])
        return fitness

    def roulette_wheel_selection(self, population, fitness):
        """Perform roulette wheel selection given the current population"""
        next_population = np.zeros(population.shape, dtype=int)
        normalized_fitness = fitness / sum(fitness)
        for i in range(0, self.population_size):
            next_population[i, :] = population[np.random.choice(population.shape[0], p=normalized_fitness), :]
        return next_population


if __name__ == "__main__":
    experiment = Experiment(trans_file="test_hmm_transition_matrix.csv",
                            emis_file="hmm_observations_emission.csv",
                            n_train_samples=2, n_test_samples=2, sample_length=20, seed=1)
    experiment.initialize()
    possible_states = [list(x) for x in list(product(np.arange(4), np.arange(4)))]
    filter = ParticleFilter(experiment)
    print(experiment.y_test)
    filter.run()
    print(filter.y_pred)
