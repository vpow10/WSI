from random import random, randint
import numpy as np
from copy import deepcopy


def get_random_genotype(size):
    return np.random.randint(0, 2, size=size)


class Individual:
    def __init__(self, genotype: list) -> None:
        self.size = len(genotype)
        self._genotype = genotype

    def get_genotype(self) -> list:
        return self._genotype

    def set_genotype(self, new_genotype: list) -> None:
        self._genotype = new_genotype

    def fitness(self, fitness_function) -> int:
        return fitness_function(self._genotype)

    def mutate(self, pm: float) -> None:
        new_genotype = []
        for gen in self._genotype:
            chance = random()
            if chance <= pm:
                gen = int(not bool(gen))
            new_genotype.append(gen)
        self.set_genotype(np.array(new_genotype))

    def crossover(self, pc: float, parent: object) -> None:
        genotype_1 = self.get_genotype()
        genotype_2 = parent.get_genotype()
        chance = random()
        if chance <= pc:
            crossover_point = randint(0, self.size - 2)
            temp = deepcopy(genotype_1)
            genotype_1[crossover_point + 1:] = genotype_2[-self.size + crossover_point + 1:]
            genotype_2[crossover_point + 1:] = temp[-self.size + crossover_point + 1:]
            self.set_genotype(genotype_1)
            parent.set_genotype(genotype_2)
