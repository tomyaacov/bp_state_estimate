from algorithm import Algorithm
from experiment import Experiment
import numpy as np
from hmm_split import *
from itertools import product


class GeneticFilter(Algorithm):
    def __init__(self, experiment,
                 possible_states,
                 population_size=5,
                 num_of_sub_processes=2,
                 genetic_operation_resolution=3,
                 mutation_probability=0.3,
                 crossover_mating_pool_size=2,
                 offspring_ratio_next_generation=0.2):
        super().__init__(experiment)
        self.possible_states = possible_states
        self.population_size = population_size
        self.num_of_sub_processes = num_of_sub_processes
        self.genetic_operation_resolution = genetic_operation_resolution
        self.mutation_probability = mutation_probability
        self.crossover_mating_pool_size = crossover_mating_pool_size
        self.offspring_ratio_next_generation = offspring_ratio_next_generation

    def run(self):
        self.y_pred = np.zeros(self.experiment.y_test.shape, dtype=int)
        for i in range(self.experiment.y_test.shape[0]):
            population = {0: self.initialize_population()}
            for j in range(self.experiment.y_test.shape[1]):

                if j%self.genetic_operation_resolution==0 and j > 0:
                    fitness = self.cal_pop_fitness_over_time(self.experiment.hmm,
                                                             [population[x] for x in range(j - self.genetic_operation_resolution + 1, j + 1)],
                                                             [self.experiment.X_test[i, x] for x in range(j - self.genetic_operation_resolution + 1, j + 1)])
                elif j==0:
                    fitness = self.cal_pop_fitness(self.experiment.hmm, population[j], self.experiment.X_test[i, j])
                else:
                    fitness = self.cal_pop_fitness_over_time(self.experiment.hmm,
                                                             [population[x] for x in range(j-j%self.genetic_operation_resolution+1, j + 1)],
                                                             [self.experiment.X_test[i, x] for x in range(j-j%self.genetic_operation_resolution+1, j + 1)])
                # state estimation
                self.y_pred[i,j] = many_to_one(population[j][np.argmax(fitness),:])
                # genetic operations
                if j % self.genetic_operation_resolution == 0 and j > 0:
                    population[j] = self.do_genetic_operation(population[j], fitness, self.experiment.X_test[i, j])
                # sampling
                population[j+1] = self.transition_population(population[j])

    def transition_population(self, population):
        next_population = np.zeros(population.shape, dtype=int)
        for i in range(population.shape[0]):
            next_population[i, :] = self.transition_sample(population[i, :])
        return next_population

    def transition_sample(self, individual):
        return one_to_many(np.random.choice(self.experiment.hmm.transmat_.shape[1], p=self.experiment.hmm.transmat_[many_to_one(individual),:]))

    def initialize_population(self):
        pop_size = (self.population_size, self.num_of_sub_processes)
        new_population = [one_to_many(np.random.choice(len(self.experiment.hmm.startprob_), p=self.experiment.hmm.startprob_)) for _ in range(self.population_size)]
        return np.reshape(new_population, pop_size)

    def cal_pop_fitness(self, hmm, pop, observation):
        fitness = np.array([hmm.emissionprob_[many_to_one(x), observation] for x in pop])
        return fitness

    def cal_pop_fitness_over_time(self, hmm, pop_list, observation_list):
        fitness = np.ones(pop_list[0].shape[0], dtype=float)
        for i in range(len(pop_list)):
            fitness *= self.cal_pop_fitness(hmm, pop_list[i], observation_list[i])
        return fitness

    def one_point_mutation(self, offspring_crossover):
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(offspring_crossover.shape[0]):
            if np.random.random() < self.mutation_probability:
                while True:
                    point = np.random.randint(1, offspring_crossover.shape[1])
                    value = np.random.randint(4)  # TODO: need to be drawn from hmm
                    if list(offspring_crossover[idx, :point])+[value]+list(offspring_crossover[idx, (point+1):]) in self.possible_states:
                        offspring_crossover[idx, point] = value
                        break
        return offspring_crossover

    def one_point_crossover(self, parents):  # TODO: need to be adjusted to more than 2 parents
        offsprings = np.zeros(parents.shape, dtype=int)
        for i in range(10):  # TODO: timeout
            point = np.random.randint(1, parents.shape[1])
            offsprings[0, :] = list(parents[0, :point]) + list(parents[1, point:])
            offsprings[1, :] = list(parents[1, :point]) + list(parents[1, point:])
            if list(offsprings[0, :]) in self.possible_states and list(offsprings[1, :]) in possible_states:
                return offsprings
        return None

    def select_mating_pool(self, population, fitness):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.zeros((self.crossover_mating_pool_size, population.shape[1]), dtype=int)
        i = 0
        for parent_num in fitness.argsort()[-self.crossover_mating_pool_size:][::-1]:
            parents[i, :] = population[parent_num, :]
            i += 1
        return parents

    def roulette_wheel_selection(self, population, offsprings, fitness, offsprings_fitness):
        next_population = np.zeros(population.shape, dtype=int)
        offspring_number = int(self.offspring_ratio_next_generation*self.population_size)
        normalized_fitness = fitness/sum(fitness)
        normalized_offsprings_fitness = offsprings_fitness / sum(offsprings_fitness)
        for i in range(offspring_number):
            next_population[i, :] = offsprings[np.random.choice(offsprings.shape[0], p=normalized_offsprings_fitness), :]
        for i in range(offspring_number, self.population_size):
            next_population[i, :] = population[np.random.choice(population.shape[0], p=normalized_fitness), :]
        return next_population

    def do_genetic_operation(self, population, fitness, current_observation):
        parents = self.select_mating_pool(population, fitness)
        offsprings = self.one_point_crossover(parents)
        offsprings = self.one_point_mutation(offsprings)
        offsprings_fitness = self.cal_pop_fitness(self.experiment.hmm, offsprings, current_observation)
        new_population = self.roulette_wheel_selection(population, offsprings, fitness, offsprings_fitness)
        return new_population




if __name__ == "__main__":
    experiment = Experiment(trans_file="test_hmm_transition_matrix.csv",
                            emis_file="hmm_observations_emission.csv",
                            n_train_samples=2, n_test_samples=2, sample_length=20, seed=1)
    experiment.initialize()
    possible_states = [list(x) for x in list(product(np.arange(4),np.arange(4)))]
    filter = GeneticFilter(experiment, possible_states)
    print(experiment.y_test)
    filter.run()
    print(filter.y_pred)
