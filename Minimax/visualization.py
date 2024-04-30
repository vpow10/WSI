import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv


def make_heatmap(path_to_file: str) -> None:
    df = pd.read_csv(path_to_file)
    df = pd.pivot_table(data=df, index='Depth2', columns='Depth1', values='Max')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, annot=True, cmap='coolwarm', vmin=0, vmax=50, square=True, linewidths=0.01, linecolor='black', ax=ax)
    ax.set_title('Max scores for different depths')
    plt.show()

def make_plot(path_to_file: str) -> None:
    data = []
    with open(path_to_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(float(row['Max_point_avg']))
    x = np.array([f'({i}, {j})' for i in range(0, 5) for j in range(0, 5)])
    plt.plot(x, data)
    plt.title('Max score average for different depths')
    plt.show()

make_heatmap('WSI/Minimax/results3.csv')
make_heatmap('WSI/Minimax/results4.csv')
make_plot('WSI/Minimax/results3.csv')