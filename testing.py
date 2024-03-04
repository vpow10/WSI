from solver import GradientIndent
import numpy as np
from random import randint


# functions and gradients

f = lambda x: 2*x**2 + 3*x - 1
f_gradient = lambda x: 4*x + 3
g = lambda x1, x2: 1 - 0.6*np.exp(- x1**2 - x2**2 ) - 0.4*np.exp( - (x1+1.75)**2 - (x2-1)**2)
g_gradient = lambda x1, x2: (1.2*x1*np.exp(- x1**2 - x2**2) + 0.8*(x1 + 1.75)*np.exp(-(x1 + 1.75)**2 - (x2-1)**2),
                              1.2*x2*np.exp(- x1**2 - x2**2) + 0.8*(x2 - 1)*np.exp(-(x1 + 1.75)**2 - (x2-1)**2))

# function for getting random numbers

def get_random(ammount, data):
    pass

# beta = 0.1

solver1 = GradientIndent(0.1)

# testing f function

