import numpy as np
# lychanl

class GradientIndent():
    def __init__(self, beta: int):
        self.beta = beta

    def get_parameters(self):
        return self.beta

    def solver(self, problem, gradient, epsilon, *args, show_data=True):
        it = 0
        point = np.array([*args])
        gradient_value = np.array(gradient(*point))
        while any(abs(value) > epsilon for value in gradient_value):
            point = point - self.beta * gradient_value
            gradient_value = np.array(gradient(*point))
            it += 1
        if show_data:
            print("--------------------")
            print("Found minimum at: ", *point)
            print("Minimum value: ", problem(*point))
            print("Iterations: ", it)
            print("--------------------")
        return (point, it)
