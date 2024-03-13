from abc import ABC, abstractmethod
from random import shuffle, choices
from fitness_function import evaluate
from individual import Individual, get_random_genotype
import numpy as np


class GeneticAlgorithm(ABC):
    def __init__(self, pc: float, pm: float) -> None:
        self._pc = pc
        self._pm = pm

    # @abstractmethod
    def get_parameters(self) -> dict:
        params = {
            'Crossover probability': self._pc,
            'Mutation probability': self._pm,
        }
        return params

    def get_fitness_sum(self, population: list, fitness_function) -> int:
        sum = 0
        for individual in population:
            sum += individual.fitness(fitness_function)
        return sum

    def get_best(self, population: list, fitness_function) -> tuple:
        best_x = None
        best_f = 0
        for individual in population:
            fitness = individual.fitness(fitness_function)
            if fitness > best_f:
                best_x = individual
                best_f = fitness
        return (best_x, best_f)

    def crossover(self, population: list) -> list:
        new_pop = []
        shuffle(population)
        pairs = np.array_split(population, len(population)/2)
        for pair in pairs:
            pair[0].crossover(self._pc, pair[1])
            new_pop.append(pair[0])
            new_pop.append(pair[1])
        return new_pop

    def mutation(self, population: list) -> list:
        new_pop = []
        for individual in population:
            individual.mutate(self._pm)
            new_pop.append(individual)
        return new_pop

    def selection(self, population: list, fitness_function) -> list:
        size = len(population)
        sum = self.get_fitness_sum(population, fitness_function)

        chances = [(individual.fitness(fitness_function)/sum)*100 for individual in population]
        return choices(population, weights=chances, k=size)

    # @abstractmethod
    def solve(self, problem, pop0: list, t_max: int, more_data=False) -> tuple:
        t = 0
        all_best_f = []
        x_best, f_best = self.get_best(pop0, problem)
        all_best_f.append(f_best)
        population = pop0
        while t < t_max:
            population = self.selection(population, problem)
            population = self.crossover(population)
            population = self.mutation(population)
            x_t, f_t = self.get_best(population, problem)
            if f_t > f_best:
                x_best = x_t
                f_best = f_t
            t += 1
            all_best_f.append(f_best)
        if more_data:
            return x_best.get_genotype(), f_best, all_best_f
        return x_best.get_genotype(), f_best


population = []
solver = GeneticAlgorithm(0.1, 0.8)
for i in range(5):
    population.append(Individual(get_random_genotype(100)))
solver.solve(evaluate, population, 100, True)
