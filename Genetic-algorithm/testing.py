from fitness_function import evaluate
from individual import Individual, get_random_genotype
import numpy as np
import matplotlib.pyplot as plt
from solver import GeneticAlgorithm


population = []
solver = GeneticAlgorithm(0.9, 0.11)
for i in range(1000):
    population.append(get_random_genotype(100))
x_best, f_best, all_f = (solver.solve(evaluate, population, 2000, True))
t_span = np.arange(2000)
print(x_best, f_best)
plt.plot(t_span, all_f)
plt.show()
