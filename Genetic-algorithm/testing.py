from fitness_function import evaluate
from individual import Individual, get_random_genotype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from solver import GeneticAlgorithm
import csv
from operator import itemgetter


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

# population = []
# t_max = 10000
# solver = GeneticAlgorithm(0.9, 0.11)
# for i in range(10000):
#     population.append(get_random_genotype(100))
# x_best, f_best, all_f, avg_f = (solver.solve(evaluate, population, t_max, True))
# t_span = np.arange(t_max)
# fig1 = plt.figure("Best fitness")
# plt.plot(t_span, all_f)
# plt.show()
# fig2 = plt.figure("Average fitness in population")
# plt.plot(t_span[::10], avg_f[::10])
# plt.show()

# x_best = [1,1,1,1,1,1,1,1,1,1,
#           1,1,1,1,1,1,1,1,1,1,
#           1,0,0,0,0,0,0,0,0,0,
#           1,1,1,1,1,1,1,1,1,1,
#           0,0,0,0,1,1,1,1,1,1,
#           1,1,1,1,1,1,0,0,0,0,
#           1,1,1,1,1,1,1,1,1,1,
#           1,0,0,0,0,0,0,0,0,0,
#           1,1,1,1,1,1,1,1,1,1,
#           1,1,1,1,1,1,1,1,1,1]
# print(evaluate(x_best))

# making a heatmap

# x_best = adjust_data(x_best)
# cmap = sns.color_palette(['white', 'red', 'green'], as_cmap=True)

# sns.heatmap(x_best, cmap=cmap, annot=True, fmt='.0f', cbar=False, vmin=0, vmax=1)
# plt.show()


# testing 200 small populations

def test_50x200():
    t_max = 5000
    size_of_x = 100
    size_of_pop = 50
    solver = GeneticAlgorithm(0.75, 0.02)
    with open('data_50x200.csv', 'w') as fh:
        f_writer = csv.writer(fh)
        f_writer.writerow(['Best fitness', 'Genotype'])
        for i in range(200):
            print(f'Testing population number {i}')
            population = [get_random_genotype(size_of_x) for _ in range(size_of_pop)]
            x_best, f_best = solver.solve(evaluate, population, t_max)
            f_writer.writerow([f_best, x_best])


def test_parameter(param: bool):
    if param:    # testing pc
        parameter = np.arange(0, 1.02, 0.02)
    else:        # testing pm
        parameter = np.arange(0, 1.02, 0.02)
    t_max = 5000
    size_of_x = 100
    size_of_pop = 50
    all_bests = []
    avgs = []
    sds = []
    for p in parameter:
        p = round(p, 2)
        print(f'testing {p}')
        if param:
            solver = GeneticAlgorithm(p, 1/size_of_pop)
        else:
            solver = GeneticAlgorithm(0.75, p)
        population = [get_random_genotype(size_of_x) for _ in range(size_of_pop)]
        x_best, f_best, all_f, _ = solver.solve(evaluate, population, t_max, True)
        all_bests.append((x_best, f_best))
        avgs.append(np.mean(all_f))
        sds.append(np.std(all_f))

    # visualizing data
    if param:
        px = 'pc'
    else:
        px = 'pm'

    fig, axs = plt.subplots(2)
    fig.suptitle(f"Testing {px} in range [0; 1]")
    axs[0].plot(parameter, avgs, '-bo')
    axs[0].set_title("Mean of best fitnesses")
    axs[1].plot(parameter, sds, '-ro')
    axs[1].set_title("Standard deviation of best fitnesses")
    plt.show()

    # making a heatmap of best individual

    x, f = max(all_bests, key=itemgetter(1))
    print(f)
    x = adjust_data(x)
    cmap = sns.color_palette(['white', 'red', 'green'], as_cmap=True)

    sns.heatmap(x, cmap=cmap, annot=True, fmt='.0f', cbar=False, vmin=0, vmax=1)
    plt.show()

# test_parameter(True)
# test_parameter(False)

test_50x200()