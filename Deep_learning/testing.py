from data_prep import prep_data
from solver import Solver
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt



def test_XOR():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    solver = Solver(2, [3], 1, 'sigmoid', 'mse', 1)
    solver.train(X, y, 40000)
    print(solver.predict(X))

def test_MNIST():
    X_mnist_train_scaled, y_mnist_train, X_mnist_val_scaled, y_mnist_val, X_mnist_test_scaled, y_mnist_test = prep_data()
    input_size_mnist = X_mnist_train_scaled.shape[1]
    hidden_sizes_mnist = [128, 64]
    output_size_mnist = 10
    learning_rate_mnist = 0.1
    epochs_mnist = 1000

    model_mnist = Solver(input_size_mnist, hidden_sizes_mnist, output_size_mnist, 'sigmoid', 'mse', learning_rate_mnist)
    model_mnist.train(X_mnist_train_scaled, y_mnist_train, epochs_mnist)

    predictions_mnist = model_mnist.predict(X_mnist_test_scaled)
    accuracy_mnist = accuracy_score(y_mnist_test, predictions_mnist)
    print(f"Test Accuracy on MNIST: {accuracy_mnist}")


    plt.figure(figsize=(10, 6))
    for i in range(10):
        index = np.random.randint(0, len(X_mnist_test_scaled))
        image = X_mnist_test_scaled[index].reshape(8, 8)
        predicted_label = predictions_mnist[index]
        true_label = y_mnist_test[index]
        plt.subplot(2, 5, i+1)
        plt.imshow(image, cmap='binary')
        plt.title(f"Predicted: {predicted_label}\nTrue: {true_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_XOR()
    test_MNIST()

