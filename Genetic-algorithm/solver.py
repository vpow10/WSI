from abc import ABC, abstractmethod
from random import random, choices
from fitness_function import evaluate
from individual import Individual, get_random_genotype


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

    # @abstractmethod
    def solve(self, problem, pop0: int, t_max: int):
        pass

    def selection(self, population: list, fitness_function) -> list:
        size = len(population)
        # evaluate sum of all fitnesses
        sum = 0
        for individual in population:
            sum += individual.fitness(fitness_function)
        # perform selection
        chances = [(individual.fitness(fitness_function)/sum)*100 for individual in population]
        return choices(population, weights=chances, k=size)


population = []
solver = GeneticAlgorithm(0.1, 0.8)
for i in range(5):
    population.append(Individual(get_random_genotype(100)))
print(population)
print(solver.selection(population, evaluate))
