from fitness_function import evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from solver import GeneticAlgorithm
import csv
from operator import itemgetter

def get_random_genotype(size):
    return np.random.randint(0, 2, size=size)

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

# testing 200 small populations

def test_populations(size_of_pop, size_of_x, pc, pm, number, t_max):
    t_span = np.arange(number)
    all_bests = []
    avgs = []
    sds = []
    solver = GeneticAlgorithm(pc, pm)
    with open('data_50x200.csv', 'w') as fh:
        f_writer = csv.writer(fh)
        f_writer.writerow(['Best fitness', 'Genotype'])
        for i in range(number):
            print(f'Testing population number {i}')
            population = [get_random_genotype(size_of_x) for _ in range(size_of_pop)]
            x_best, f_best, all_f, _ = solver.solve(evaluate, population, t_max, True)
            all_bests.append((x_best, f_best))
            avgs.append(np.mean(all_f))
            sds.append(np.std(all_f))
            f_writer.writerow([f_best, x_best])
    fig, axs = plt.subplots(2)
    fig.suptitle(f"Testing {number} populations with pc = {pc}; pm = {pm}")
    axs[0].plot(t_span, avgs, '-bo')
    axs[0].set_title("Mean of best fitnesses")
    axs[1].plot(t_span, sds, '-ro')
    axs[1].set_title("Standard deviation of best fitnesses")
    plt.show()

    fig2 = plt.figure("Best fitness over time")
    plt.plot(np.arange(t_max), all_f)
    plt.xlabel('Time')
    plt.ylabel('Best fitness')
    plt.show()

    # making a heatmap of best individual

    x, f = max(all_bests, key=itemgetter(1))
    print(f)
    x = adjust_data(x)
    cmap = sns.color_palette(['white', 'red', 'green'], as_cmap=True)

    sns.heatmap(x, cmap=cmap, annot=True, fmt='.0f', cbar=False, vmin=0, vmax=1)
    plt.show()


def test_parameter(param: bool):
    if param:    # testing pc
        parameter = np.arange(0, 1.02, 0.02)
    else:        # testing pm
        parameter = np.arange(0, 1.02, 0.02)
    t_max = 2000
    size_of_x = 100
    size_of_pop = 50
    all_bests = []
    avgs_best = []
    avgs = []
    sds = []
    sds_best = []
    for p in parameter:
        p = round(p, 2)
        print(f'testing {p}')
        if param:
            solver = GeneticAlgorithm(p, 1/size_of_pop)
        else:
            solver = GeneticAlgorithm(0.8, p)
        population = [get_random_genotype(size_of_x) for _ in range(size_of_pop)]
        x_best, f_best, all_f, all_avgs = solver.solve(evaluate, population, t_max, True)
        all_bests.append((x_best, f_best))
        avgs_best.append(np.mean(all_f))
        sds_best.append(np.std(all_f))
        avgs.append(np.mean(all_avgs))
        sds.append(np.std(all_avgs))

    # visualizing data
    if param:
        px = 'pc'
    else:
        px = 'pm'


    fig, axs = plt.subplots(2)
    fig.suptitle(f"Testing {px} in range [0; 1]")
    axs[0].plot(parameter, avgs_best, '-bo')
    axs[0].set_title("Mean of best fitnesses")
    axs[1].plot(parameter, sds_best, '-ro')
    axs[1].set_title("Standard deviation of best fitnesses")
    plt.show()

    fig2, axs2 = plt.subplots(2)
    fig2.suptitle(f"Testing {px} in range [0; 1]")
    axs2[0].plot(parameter, avgs_best, '-bo')
    axs2[0].set_title("Mean of all fitnesses")
    axs2[1].plot(parameter, sds_best, '-ro')
    axs2[1].set_title("Standard deviation of all fitnesses")
    plt.show()

    # making a heatmap of best individual

    x, f = max(all_bests, key=itemgetter(1))
    print(f)
    x = adjust_data(x)
    cmap = sns.color_palette(['white', 'red', 'green'], as_cmap=True)

    sns.heatmap(x, cmap=cmap, annot=True, fmt='.0f', cbar=False, vmin=0, vmax=1)
    plt.show()

test_parameter(True)
# test_parameter(False)

# test_populations(50, 100, 0.8, 0.02, 20, 1000)