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


def test_epoch_impact():
    X_mnist_train_scaled, y_mnist_train, X_mnist_val_scaled, y_mnist_val, _, _ = prep_data()
    input_size_mnist = X_mnist_train_scaled.shape[1]
    hidden_sizes_mnist = [128, 64]
    output_size_mnist = 10
    learning_rate_mnist = 2
    accuracy = []
    losses = []
    for epochs in range(100, 2100, 100):
        model_mnist = Solver(input_size_mnist, hidden_sizes_mnist, output_size_mnist, 'sigmoid', 'mse', learning_rate_mnist)
        loss = model_mnist.train(X_mnist_train_scaled, y_mnist_train, epochs)
        predictions_mnist = model_mnist.predict(X_mnist_val_scaled)
        accuracy_mnist = accuracy_score(y_mnist_val, predictions_mnist)
        accuracy.append(accuracy_mnist)
        losses.append(loss)
        print(f"Validation Accuracy on MNIST with {epochs} epochs: {accuracy_mnist}")
    epochs = list(range(100, 2100, 100))
    plt.figure(1)
    plt.plot(epochs, accuracy, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Epochs')
    plt.grid(True)
    plt.figure(2)
    plt.plot(epochs, losses, marker='o', linestyle='-', color='r')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Number of Epochs')
    plt.grid(True)
    plt.show()


def test_lr_impact():
    X_train, y_train, X_val, y_val, _, _ = prep_data()
    input_size = X_train.shape[1]
    hidden_sizes = [128, 64]
    output_size = 10
    learning_rate = np.arange(0.1, 2.1, 0.1)
    epochs = 1000
    accuracies = []
    losses = []
    for lr in learning_rate:
        model = Solver(input_size, hidden_sizes, output_size, 'sigmoid', 'mse', lr)
        loss = model.train(X_train, y_train, epochs)
        losses.append(loss)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        accuracies.append(accuracy)
        print(f"Validation Accuracy on MNIST with lr={lr}: {accuracy}")
    plt.figure(1)
    plt.plot(learning_rate, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Learning Rate Impact on Accuracy')
    plt.grid(True)
    plt.figure(2)
    plt.plot(learning_rate, losses, marker='o', linestyle='-', color='r')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Impact on Loss')
    plt.grid(True)
    plt.show()

def test_hidden_size_impact():
    X_train, y_train, X_val, y_val, _, _ = prep_data()
    input_size = X_train.shape[1]
    output_size = 10
    sizes = [[3], [3, 3], [10, 10], [30, 60], [128, 64], [30, 30, 30]]
    epochs = 1000
    lr = 0.1
    accuracies = []
    losses = []
    for hidden_size in sizes:
        model = Solver(input_size, hidden_size, output_size, 'sigmoid', 'mse', lr)
        loss = model.train(X_train, y_train, epochs)
        losses.append(loss)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        accuracies.append(accuracy)
        print(f"Validation Accuracy on MNIST with: {accuracy}")
    plt.figure(1)
    plt.plot(range(len(sizes)), accuracies, marker='o', linestyle='-', color='b')
    plt.xticks(range(len(sizes)), [str(size) for size in sizes])
    plt.xlabel('Hidden Layer Sizes')
    plt.ylabel('Accuracy')
    plt.title('Hidden Layer Sizes Impact on Accuracy')
    plt.grid(True)
    plt.figure(2)
    plt.plot(range(len(sizes)), losses, marker='o', linestyle='-', color='r')
    plt.xticks(range(len(sizes)), [str(size) for size in sizes])
    plt.xlabel('Hidden Layer Sizes')
    plt.ylabel('Loss')
    plt.title('Hidden Layer Sizes Impact on Loss')
    plt.grid(True)
    plt.show()

def test_MNIST():
    X_mnist_train_scaled, y_mnist_train, X_mnist_val_scaled, y_mnist_val, X_mnist_test_scaled, y_mnist_test = prep_data()
    input_size_mnist = X_mnist_train_scaled.shape[1]
    hidden_sizes_mnist = [128, 64]
    output_size_mnist = 10
    learning_rate_mnist = 2
    epochs_mnist = 2000

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
    # test_XOR()
    # test_epoch_impact()
    # test_lr_impact()
    # test_hidden_size_impact()
    test_MNIST()