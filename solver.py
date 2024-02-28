import numpy as np


class GradientIndent():
    def __init__(self, beta: int):
        self.beta = beta

    def get_parameters(self):
        return self.beta

    def solver(self, problem, gradient, epsilon, *args):
        it = 0
        point = np.array(args)
        if len(point) == 1:
            while abs((gradient(*point))) > epsilon:
                point = point - self.beta * gradient(point)
                it += 1
            return [problem(*point), it]
        else:
            while not self.check_slope(problem, epsilon, point):
                point, more_it = self.slope_direction(problem, epsilon, point)
                it += more_it
            while any(abs(value) > epsilon for value in gradient(*point)):
                point = point - self.beta * np.array(gradient(*point))
                it += 1
            return [round(problem(*point),4), it]

    def check_slope(self, problem, epsilon, variables):
        point = np.array(variables)
        if abs(problem(*point) - problem(*(point - 1))) <= epsilon:
            return False
        else:
            return True

    def slope_direction(self, problem, epsilon, variables):
        start_point = np.array(variables)
        next_point = np.array(start_point + 1)
        it = 0
        dx = np.diff((problem(*start_point), problem(*(next_point))))
        print(dx)
        if dx >= 0 and problem(*start_point) > 0:
            next_point -= 2
            while abs(np.diff((problem(*next_point), problem(*start_point)))) < epsilon:
                it += 1
                next_point -= 1
        else:
            while abs(np.diff((problem(*start_point), problem(*next_point)))) < epsilon:
                it += 1
                next_point += 1
        return [next_point, it]


f = lambda x: 2*x**2 + 3*x - 1
f_gradient = lambda x: 4*x + 3
g = lambda x1, x2: 1 - 0.6*np.exp(- x1**2 - x2**2 ) - 0.4*np.exp( - (x1+1.75)**2 - (x2-1)**2)
g_gradient = lambda x1, x2: (1.2*x1*np.exp(- x1**2 - x2**2) + 0.8*(x1 + 1.75)*np.exp(-(x1 + 1.75)**2 - (x2-1)**2),
                              1.2*x2*np.exp(- x1**2 - x2**2) + 0.8*(x2 - 1)*np.exp(-(x1 + 1.75)**2 - (x2-1)**2))

a = GradientIndent(0.25)
print(a.solver(f, f_gradient, 0.00001, 500000000000000000))
print(a.solver(g, g_gradient, 0.00001, -3, -2))
print(g(-5, -5))
print(g(-4, -4))
# print(a.check_slope(g, 0.00001, -1, -1))