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
betas = np.arange(0.02, 0.49, 0.01)

# getting data for heatmap and making a csv file for f function

def get_data_f():
    starting_points = [np.array([point]) for point in np.arange(-100, 105, 5)]
    all_its = []
    for beta in betas:
        solver = GradientIndent(beta)
        its = []
        for point in starting_points:
            its.append(solver.solver(f, f_gradient, epsilon, point, show_data=False)[1])
        all_its.append(its)

    with open('Gradient_descent/data_f.csv', 'w') as fh:
        f_writer = csv.writer(fh)
        f_writer.writerow(['Beta', 'Starting point', 'Iterations'])
        for n in range(len(betas)):
            for i in range(len(starting_points)):
                f_writer.writerow([round(betas[n], 2), starting_points[i][0], all_its[n][i]])

# making a heatmap for f function

get_data_f()
df = pd.read_csv('Gradient_descent/data_f.csv')
df = pd.pivot_table(data=df, index="Beta", columns="Starting point", values="Iterations")
fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(df, cmap='coolwarm', square=True, vmin=1, vmax=50, fmt=',.0f', linewidths=.01)
ax.set_title('Iterations for each beta over starting points from (-100, 100)')
plt.show()
