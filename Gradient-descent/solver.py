import numpy as np


class GradientIndent():
    def __init__(self, beta: int):
        self.beta = beta

    def get_parameters(self):
        return self.beta

    def solver(self, problem, gradient, epsilon, *args, show_data=True):
        it = 0
        point = np.array(args)
        if len(point) == 1:
            while abs((gradient(*point))) > epsilon:
                point = point - self.beta * gradient(point)
                it += 1
        else:
            # while not self.check_slope(problem, epsilon, point):
            #     point, more_it = self.slope_direction(problem, epsilon, point)
            #     it += more_it
            while any(abs(value) > epsilon for value in gradient(*point)):
                point = point - self.beta * np.array(gradient(*point))
                it += 1
        if show_data:
            print("--------------------")
            print("Found minimum at: ", *point)
            print("Minimum value: ", problem(*point))
            print("Iterations: ", it)
            print("--------------------")
        return (point, it)

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
