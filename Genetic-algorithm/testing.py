from fitness_function import evaluate
from individual import Individual, get_random_genotype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from solver import GeneticAlgorithm


def adjust_data(x):
    n = int(len(x) ** (0.5))
    checked = [[False] * n for _ in range(n)]
    avaliable = [[False] * n for _ in range(n)]

    to_check = []

    for i in range(n): # accessible from outside
        to_check.append((i, 0))
        to_check.append((i, n - 1))
        to_check.append((0, i))
        to_check.append((n - 1, i))

    while to_check:
        i, j = to_check.pop()

        if checked[i][j]:
            continue

        checked[i][j] = True
        if x[i * n + j]: # parking spot
            avaliable[i][j] = True
        else: # road
            if i > 0:
                to_check.append((i - 1, j))
            if i < n - 1:
                to_check.append((i + 1, j))
            if j > 0:
                to_check.append((i, j - 1))
            if j < n - 1:
                to_check.append((i, j + 1))

    for i in range(n):
        for j in range(n):
            # print(avaliable[i][j], x[i*n+j])
            if not avaliable[i][j] and x[i*n+j]:
                x[i*n+j] = 0.6
    x = np.array_split(x, n)
    return x

population = []
solver = GeneticAlgorithm(0.95, 0.09)
for i in range(1000):
    population.append(get_random_genotype(100))
x_best, f_best, all_f = (solver.solve(evaluate, population, 2000, True))
t_span = np.arange(2000)
fig1 = plt.figure("Best fitness")
plt.plot(t_span, all_f)
plt.show()

# making a heatmap
x_best = adjust_data(x_best)
cmap = sns.color_palette(['white', 'red', 'green'], as_cmap=True)

sns.heatmap(x_best, cmap=cmap, annot=True, fmt='.0f', cbar=False, vmin=0, vmax=1)
plt.show()