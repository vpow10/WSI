from fitness_function import evaluate
from individual import Individual, get_random_genotype
import numpy as np
import matplotlib.pyplot as plt
from solver import GeneticAlgorithm


# population 100, t_max 10000
population = []
for _ in range(100):
    population.append(Individual(get_random_genotype(100)))
solver = GeneticAlgorithm(0.75, 0.05)
t_span = np.arange(0, 10001, 1)
x_best, f_best, all_f = solver.solve(evaluate, population, 10000, True)
plt.plot(t_span, all_f)
print(x_best, f_best)
plt.show()
