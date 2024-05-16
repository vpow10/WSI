import numpy as np
from typing import List


class Solver:
    def __init__(
            self, input_size: int, hidden_sizes: List[int] , output_size: int,
            activation_function: str, loss_function: str, learning_rate: float):
        """
        Hidden sizes is a list of integers, each representing the number of neurons in a hidden layer
        Possible activation functions: 'sigmoid', 'tanh', 'relu'
        Possible loss functions: 'mse', 'cross_entropy'
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.hidden_number = len(hidden_sizes)
        self.output_size = output_size
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate

        # Initialize weights and biases

        self.W1 = np.random.randn(self.input_size, self.hidden_sizes[0])
        self.W_hidden = [np.random.randn(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(self.hidden_number - 1)]
        self.W2 = np.random.randn(self.hidden_sizes[-1], self.output_size)
        self.b1 = np.zeros((1, self.hidden_sizes[0]))
        self.b_hidden = [np.zeros((1, self.hidden_sizes[i])) for i in range(self.hidden_number - 1)]
        self.b2 = np.zeros((1, self.output_size))

    def activation(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)

    def derivative(self, x):
        if self.activation_function == 'sigmoid':
            return self.activation(x) * (1 - self.activation(x))
        elif self.activation_function == 'tanh':
            return 1 - self.activation(x)**2
        elif self.activation_function == 'relu':
            return 1 if x > 0 else 0

    def loss(self, y_true, y_pred):
        if self.loss_function == 'mse':
            return np.mean((y_true - y_pred)**2)
        elif self.loss_function == 'cross_entropy':
            return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        if self.hidden_number == 1:
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.activation(self.z2)
            return self.a2
        self.z_hidden = [np.dot(self.a1, self.W_hidden[0]) + self.b_hidden[0]]
        self.a_hidden = [self.activation(self.z_hidden[0])]
        for i in range(1, self.hidden_number - 1):
            self.z_hidden.append(np.dot(self.a_hidden[i-1], self.W_hidden[i]) + self.b_hidden[i])
            self.a_hidden.append(self.activation(self.z_hidden[i]))
        self.z2 = np.dot(self.a_hidden[-1], self.W2) + self.b2
        self.a2 = self.activation(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        self.loss_value = self.loss(y, self.a2)


        if self.hidden_number == 1:
            # Calculate gradients
            dz2 = self.a2 - y
            dW2 = np.dot(self.a1.T, dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m
            dz1 = np.dot(dz2, self.W2.T) * self.derivative(self.z1)
            dW1 = np.dot(X.T, dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m

            # Update weights and biases
            self.W1 -= self.learning_rate * dW1
            self.W2 -= self.learning_rate * dW2
            self.b1 -= self.learning_rate * db1
            self.b2 -= self.learning_rate * db2
        else:
            # Calculate gradients
            dz2 = self.a2 - y
            dW2 = np.dot(self.a_hidden[-1].T, dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m
            dz_hidden = [np.dot(dz2, self.W2.T) * self.derivative(self.z_hidden[-1])]
            dW_hidden = [np.dot(self.a_hidden[-1].T, dz_hidden[0]) / m]
            db_hidden = [np.sum(dz_hidden[0], axis=0, keepdims=True) / m]
            for i in range(self.hidden_number - 2, 0, -1):
                dz_hidden.append(np.dot(dz_hidden[-1], self.W_hidden[i].T) * self.derivative(self.z_hidden[i-1]))
                dW_hidden.append(np.dot(self.a_hidden[i-1].T, dz_hidden[-1]) / m)
                db_hidden.append(np.sum(dz_hidden[-1], axis=0, keepdims=True) / m)
            dz1 = np.dot(dz_hidden[-1], self.W_hidden[0].T) * self.derivative(self.z1)
            dW1 = np.dot(X.T, dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m

            # Update weights and biases
            self.W1 -= self.learning_rate * dW1
            self.W2 -= self.learning_rate * dW2
            for i in range(self.hidden_number - 1):
                self.W_hidden[i] -= self.learning_rate * dW_hidden[i]
                self.b_hidden[i] -= self.learning_rate * db_hidden[i]
            self.b1 -= self.learning_rate * db1
            self.b2 -= self.learning_rate * db2

        # self.loss_value = self.loss(y, self.a2)
        # self.a2_delta = y - self.a2
        # self.z2_delta = self.a2_delta * self.derivative(self.z2)
        # if self.hidden_number == 1:
        #     self.W2_delta = self.a1.T.dot(self.z2_delta)
        #     self.b2_delta = np.sum(self.z2_delta, axis=0)
        #     self.z1_delta = self.z2_delta * self.derivative(self.z1)
        #     self.W1_delta = X.T.dot(self.z1_delta)
        #     self.b1_delta = np.sum(self.z1_delta, axis=0)
        #     self.W1 += self.W1_delta * self.learning_rate
        #     self.W2 += self.W2_delta * self.learning_rate
        #     self.b1 += self.b1_delta * self.learning_rate
        #     self.b2 += self.b2_delta * self.learning_rate
        #     return
        # self.a_hidden_delta = [self.z2_delta.dot(self.W2.T)]
        # self.W2_delta = self.a_hidden[-1].T.dot(self.z2_delta)
        # self.b2_delta = np.sum(self.z2_delta, axis=0)
        # for i in range(self.hidden_number - 2, -1, -1):
        #     self.z_hidden_delta = self.a_hidden_delta[-1] * self.derivative(self.z_hidden[i])
        #     self.a_hidden_delta.append(self.z_hidden_delta.dot(self.W_hidden[0][i+1].T))
        #     self.W_hidden_delta = self.a_hidden[i-1].T.dot(self.z_hidden_delta)
        #     self.b_hidden_delta = np.sum(self.z_hidden_delta, axis=0)
        # self.z1_delta = self.a_hidden_delta[-1] * self.derivative(self.z1[:, 0])
        # self.W1_delta = X.T.dot(self.z1_delta)
        # self.b1_delta = np.sum(self.z1_delta, axis=0)

        # # Update weights and biases

        # self.W1 += self.W1_delta * self.learning_rate
        # self.W2 += self.W2_delta * self.learning_rate
        # for i in range(self.hidden_number - 1):
        #     self.W_hidden[i] += self.W_hidden_delta * self.learning_rate
        # self.b1 += self.b1_delta * self.learning_rate
        # self.b2 += self.b2_delta * self.learning_rate
        # for i in range(self.hidden_number - 1):
        #     self.b_hidden[i] += self.b_hidden_delta * self.learning_rate

    def train(self, X, y, epochs):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y)
        print(self.loss_value)

    def predict(self, X):
        return np.round(self.forward(X), 3)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

solver = Solver(2, [3, 3], 1, 'sigmoid', 'mse', 0.1)
solver.train(X, y, 50000)
print(solver.predict(X))
