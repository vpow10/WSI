from solver import GradientIndent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv


# functions and gradients

f = lambda x: 2*x**2 + 3*x - 1
f_gradient = lambda x: 4*x + 3
g = lambda x1, x2: 1 - 0.6*np.exp(- x1**2 - x2**2 ) - 0.4*np.exp( - (x1+1.75)**2 - (x2-1)**2)
g_gradient = lambda x1, x2: (1.2*x1*np.exp(- x1**2 - x2**2) + 0.8*(x1 + 1.75)*np.exp(-(x1 + 1.75)**2 - (x2-1)**2),
                              1.2*x2*np.exp(- x1**2 - x2**2) + 0.8*(x2 - 1)*np.exp(-(x1 + 1.75)**2 - (x2-1)**2))
epsilon = 1e-6
betas = np.arange(0.02, 0.48, 0.01)

# getting data for heatmap

starting_points = [np.array([point]) for point in np.arange(-100, 100, 10)]
all_its = []
for beta in betas:
    solver = GradientIndent(beta)
    its = []
    for point in starting_points:
        its.append(solver.solver(f, f_gradient, epsilon, point, show_data=False)[1])
    all_its.append(its)

# making a csv file

with open('Gradient-descent/data_f.csv', 'w') as fh:
    f_writer = csv.writer(fh)
    f_writer.writerow(['beta', 'starting point', 'iterations'])
    for n in range(len(betas)):
        for i in range(len(starting_points)):
            f_writer.writerow([round(betas[n], 2), starting_points[i][0], all_its[n][i]])

# making a heatmap

df = pd.read_csv('Gradient-descent/data_f.csv')