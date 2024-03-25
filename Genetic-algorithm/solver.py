from random import choices
from fitness_function import evaluate
import numpy as np
from random import random, randint
from copy import deepcopy


class GeneticAlgorithm():
    def __init__(self, pc: float, pm: float) -> None:
        self._pc = pc
        self._pm = pm

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
        size = len(population)
        while population:
            chance = random()
            first_parent = population.pop(0)
            second_parent = population.pop(0)
            if chance <= self._pc:
                crossover_point = randint(0, size - 1)
                first_child = np.array(list(first_parent[:crossover_point]) + list(second_parent[crossover_point:]))
                second_child = np.array(list(second_parent[:crossover_point]) + list(first_parent[crossover_point:]))
                new_pop.append(first_child)
                new_pop.append(second_child)
            else:
                new_pop.append(first_parent)
                new_pop.append(second_parent)
        return new_pop

    def mutation(self, population: list) -> list:
        for x in population:
            for n in range(len(x)):
                chance = random()
                if chance <= self._pm:
                    x[n] = int(not bool(x[n]))
        return population

    def selection(self, population: list, fitness) -> list:
        fitness_sum = sum(fitness)
        chances = [fitness_v/fitness_sum for fitness_v in fitness]
        population = choices(population, chances, k=len(population))
        return population

    def solve(self, problem, pop0: list, t_max: int, more_data=False) -> tuple:
        # checking if individuals in first population are well-formed
        # and if population doesn't consist of only zeros individuals
        pop_sum = 0
        for individual in pop0:
            pop_sum += sum(individual)
            for gene in individual:
                if gene < 0 or gene > 1:
                    return "Population is malformed."
        if pop_sum == 0:
            return "Population is malformed."
        t = 0
        all_f = []
        avg_f = []
        population = pop0
        fitness = [problem(x) for x in pop0]
        f_best = max(fitness)
        x_best = pop0[np.argmax(fitness)]
        while t < t_max:
            population = self.selection(population, fitness)
            population = self.crossover(population)
            population = self.mutation(population)
            fitness = [evaluate(x) for x in population]
            f_t = np.max(fitness)
            if f_t > f_best:
                x_best = deepcopy(population[np.argmax(fitness)])
                f_best = f_t
            all_f.append(f_best)
            avg_f.append(sum(fitness)/len(population))
            t += 1
        if more_data:
            return x_best, f_best, all_f, avg_f
        return x_best, f_best
