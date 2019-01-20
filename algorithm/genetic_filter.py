from algorithm.algorithm import Algorithm
from algorithm.particle_filter import ParticleFilter
from experiment.experiment import Experiment
from hmm_model.hmm import *
from itertools import product


class GeneticFilter(ParticleFilter):
    """Class Genetic Filter algorithm for the suggested adapted genetic filter"""
    def __init__(self,
                 name,
                 possible_states,
                 population_size=5,
                 num_of_sub_processes=2,
                 genetic_operation_resolution=3,
                 mutation_probability=0.3,
                 crossover_mating_pool_size=2,
                 offspring_ratio_next_generation=0.2):
        super().__init__(name, population_size, num_of_sub_processes)
        self.possible_states = possible_states
        self.population_size = population_size
        self.num_of_sub_processes = num_of_sub_processes
        self.genetic_operation_resolution = genetic_operation_resolution
        self.mutation_probability = mutation_probability
        self.crossover_mating_pool_size = crossover_mating_pool_size
        self.offspring_ratio_next_generation = offspring_ratio_next_generation
        self.mean_population_fitness = None

    def run(self):
        """running the algorithm for each sample"""
        self.y_pred = np.zeros(self.experiment.y_test.shape, dtype=int)
        self.mean_population_fitness = np.zeros(self.experiment.y_test.shape, dtype=float)
        for i in range(self.experiment.y_test.shape[0]):
            population = {0: self.initialize_population()}
            for j in range(self.experiment.y_test.shape[1]):
                # calculating fitness
                if j%self.genetic_operation_resolution == 0 and j > 0:
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
                # computing mean fitness
                self.mean_population_fitness[i,j] = np.mean(fitness)
                # sampling
                population[j+1] = self.transition_population(population[j])

    def one_point_mutation(self, offspring_crossover):  # need to be adjusted in case of different hmm's
        """Performs random resetting mutation. Mutation changes a single integer in each offspring randomly."""
        for idx in range(offspring_crossover.shape[0]):
            if np.random.random() < self.mutation_probability:
                while True:
                    point = np.random.randint(1, offspring_crossover.shape[1])
                    value = np.random.randint(4)  # need to be drawn from hmm
                    if list(offspring_crossover[idx, :point])+[value]+list(offspring_crossover[idx, (point+1):]) in self.possible_states:
                        offspring_crossover[idx, point] = value
                        break
        return offspring_crossover

    def one_point_crossover(self, parents):  # need to be adjusted to more than 2 parents in case of different hmm's
        """Perform one point crossover over the parents"""
        offsprings = np.zeros(parents.shape, dtype=int)
        for i in range(10):  # timeout
            point = np.random.randint(1, parents.shape[1])
            offsprings[0, :] = list(parents[0, :point]) + list(parents[1, point:])
            offsprings[1, :] = list(parents[1, :point]) + list(parents[1, point:])
            if list(offsprings[0, :]) in self.possible_states and list(offsprings[1, :]) in self.possible_states:
                return offsprings
        return None

    def select_mating_pool(self, population, fitness):
        """Selecting the best individuals in the current generation as parents for
        producing the offspring of the next generation."""
        parents = np.zeros((self.crossover_mating_pool_size, population.shape[1]), dtype=int)
        i = 0
        for parent_num in fitness.argsort()[-self.crossover_mating_pool_size:][::-1]:
            parents[i, :] = population[parent_num, :]
            i += 1
        return parents

    def roulette_wheel_selection(self, population, offsprings, fitness, offsprings_fitness):
        """Perform roulette wheel selection given the current population and it's offsprings"""
        next_population = np.zeros(population.shape, dtype=int)
        offspring_number = int(self.offspring_ratio_next_generation*self.population_size)
        normalized_fitness = fitness/sum(fitness)
        normalized_offsprings_fitness = offsprings_fitness / sum(offsprings_fitness)
        for i in range(offspring_number):  # selecting offsprings
            next_population[i, :] = offsprings[np.random.choice(offsprings.shape[0], p=normalized_offsprings_fitness), :]
        for i in range(offspring_number, self.population_size):  # selecting from current population
            next_population[i, :] = population[np.random.choice(population.shape[0], p=normalized_fitness), :]
        return next_population

    def do_genetic_operation(self, population, fitness, current_observation):
        """Perform genetic operations over population"""
        parents = self.select_mating_pool(population, fitness)
        offsprings = self.one_point_crossover(parents)
        offsprings = self.one_point_mutation(offsprings)
        offsprings_fitness = self.cal_pop_fitness(self.experiment.hmm, offsprings, current_observation)
        new_population = self.roulette_wheel_selection(population, offsprings, fitness, offsprings_fitness)
        return new_population

    def get_mean_fitness(self):
        """return the mean fitness of the algorithm"""
        return range(self.genetic_operation_resolution, self.mean_population_fitness.shape[1], self.genetic_operation_resolution), np.mean(self.mean_population_fitness[:, range(self.genetic_operation_resolution, self.mean_population_fitness.shape[1], self.genetic_operation_resolution)], axis=0)


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
