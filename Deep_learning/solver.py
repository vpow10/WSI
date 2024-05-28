import numpy as np
from sklearn.metrics import accuracy_score
from typing import List


class Solver:
    def __init__(self, input_size: int, hidden_sizes: List[int] , output_size: int,
            activation_function: str, loss_function: str, learning_rate: float):
        """
        Hidden sizes is a list of integers, each representing the number of neurons in a hidden layer
        Possible activation functions: 'sigmoid', 'tanh', 'relu'
        Possible loss functions: 'mse', 'cross_entropy'
        """
        if input_size <= 0 or output_size <= 0 or learning_rate <= 0:
            raise ValueError("Input size, output size and learning rate must be positive integers")
        if activation_function not in ['sigmoid', 'tanh', 'relu']:
            raise ValueError("Activation function must be 'sigmoid', 'tanh' or 'relu'")
        if loss_function not in ['mse', 'cross_entropy']:
            raise ValueError("Loss function must be 'mse' or 'cross_entropy'")
        if not all([size > 0 for size in hidden_sizes]):
            raise ValueError("Hidden sizes must be positive integers")
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) / np.sqrt(layer_sizes[i]))
            self.biases.append(np.random.randn(layer_sizes[i+1]))

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
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation = self.activation(z)
            activations.append(activation)
        return activations

    def predict(self, X):
        activations = self.forward(X)
        if self.output_size > 1:
            return np.argmax(activations[-1], axis=1)
        else:
            return np.round(activations[-1]).astype(int)

    def backward(self, X, y):
        activations = self.forward(X)
        output_activation = activations[-1]
        if self.output_size > 1:
            error = output_activation - np.eye(self.output_size)[y]
        else:
            error = output_activation - y.reshape(-1, 1)
        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                delta = error * self.derivative(output_activation)
            else:
                delta = np.dot(delta, self.weights[i+1].T) * self.derivative(activations[i+1])
            gradient_weights = np.dot(activations[i].T, delta) / len(X)
            gradient_biases = np.mean(delta, axis=0)
            self.weights[i] -= self.learning_rate * gradient_weights
            self.biases[i] -= self.learning_rate * gradient_biases

    def train(self, X, y, epochs):
        for epoch in range(epochs+1):
            self.backward(X, y)
            if epoch % 100 == 0:
                predictions = self.predict(X)
                loss = self.loss(y, predictions)
                accuracy = accuracy_score(y, predictions)
                print(f"Epoch {epoch}, Accuracy: {accuracy}, Loss: {loss}")
        return loss
