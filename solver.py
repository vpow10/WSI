import numpy as np


class GradientIndent():
    def __init__(self, beta: int):
        self.beta = beta

    def get_parameters(self):
        return self.beta

    def solver(self, problem, gradient, x0, epsilon, *args):
        point = x0

        while abs(gradient(point)) > epsilon:
            point = point - self.beta * gradient(point)
        return round(problem(point),4)

f = lambda x: 2*x**2 + 3*x - 1
f_gradient = lambda x: 4*x + 3
# def g(x1, x2):
#     return 1-0.6*np.exp(x1**2+x2**2)-0.4*np.exp(-(x1+1.75)**2-(x2-1)**2)
# g = lambda x1, x2: 1-0.6*np.exp(- x1**2 - x2**2 )-0.4*np.exp( - (x1+1.75)**2 - (x2-1)**2)
# g_gradient =

a = GradientIndent(0.1)
print(a.solver(f, f_gradient, -0.8, 0.001))

