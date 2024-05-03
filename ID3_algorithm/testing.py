import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from solver import ID3_algorithm, Node
from data_prep import discretize
from sklearn.metrics import classification_report, accuracy_score


def print_tree(node, depth=0):
    if node.result is not None:
        print("  " * depth, "Result:", node.result)
    else:
        print("  " * depth, "Feature:", node.feature)
        for value, child in node.children.items():
            print("  " * (depth+1), f"Value {value}:")
            print_tree(child, depth + 2)

def single_prediction(depth: int, show_values=False):
    data = pd.read_csv('data/cardio_train.csv', sep=';', index_col="id")
    data = discretize(data)
    data_train, data_val = train_test_split(data, test_size=0.2, random_state=47)
    data_val, data_test = train_test_split(data_val, test_size=0.5, random_state=47)
    Y_train = data_train['cardio']
    X_train = data_train.drop('cardio', axis=1)
    Y_val = data_val['cardio']
    X_val = data_val.drop('cardio', axis=1)
    Y_test = data_test['cardio']
    X_test = data_test.drop('cardio', axis=1)

    # Modelling and checking the accuracy

    model = ID3_algorithm(data_train)
    model.result = model.fit(X_train, Y_train, depth)
    model.predictions = model.predict(X_train, model.result)
    train_accuracy = model.accuracy(Y_train.tolist(), model.predictions)

    # Validation accuracy

    validation_predictions = model.predict(X_val, model.result)
    validation_accuracy = model.accuracy(Y_val.tolist(), validation_predictions)

    # Test accuracy

    test_predictions = model.predict(X_test, model.result)
    test_accuracy = model.accuracy(Y_test.tolist(), test_predictions)

    if show_values:
        print(f"Training accuracy: {train_accuracy}")
        print(f"Validation accuracy: {validation_accuracy}")
        print(f"Test accuracy: {test_accuracy}")
        print(f"Classification report: {classification_report(Y_test.tolist(), test_predictions)}")

    return train_accuracy, validation_accuracy, test_accuracy

def test_depths(number: int):
    train_accuracies = []
    validation_accuracies = []
    depths_accuracy = {}
    for i in range(1, number+1):
        print(f"Depth: {i}")
        train_accuracy, validation_accuracy, _ = single_prediction(i)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        depths_accuracy[i] = validation_accuracy
    fig = plt.figure("Accuracy vs Depth")
    plt.plot(range(1, number+1), train_accuracies, label="Train")
    plt.plot(range(1, number+1), validation_accuracies, label="Validation")
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.legend()

    best_depth = max(depths_accuracy, key=depths_accuracy.get)
    plt.show()
    return best_depth

def full_test():
    data = pd.read_csv('data/cardio_train.csv', sep=';', index_col="id")
    data = discretize(data)
    data_train, data_val = train_test_split(data, test_size=0.2, random_state=47)
    data_val, data_test = train_test_split(data_val, test_size=0.5, random_state=47)
    Y_train = data_train['cardio']
    X_train = data_train.drop('cardio', axis=1)
    Y_val = data_val['cardio']
    X_val = data_val.drop('cardio', axis=1)
    Y_test = data_test['cardio']
    X_test = data_test.drop('cardio', axis=1)

    # Finding the best depth for the model and testing that model
    best_depth = test_depths(20)
    model = ID3_algorithm(data_train)
    model.result = model.fit(X_train, Y_train, best_depth)
    model.predictions = model.predict(X_train, model.result)
    train_accuracy = model.accuracy(Y_train.tolist(), model.predictions)

    # Validation accuracy
    validation_predictions = model.predict(X_val, model.result)
    validation_accuracy = model.accuracy(Y_val.tolist(), validation_predictions)

    # Test accuracy
    test_predictions = model.predict(X_test, model.result)
    test_accuracy = model.accuracy(Y_test.tolist(), test_predictions)

    print(f"Best depth: {best_depth}")
    print(f"Training accuracy: {round(train_accuracy, 3)}")
    print(f"Validation accuracy: {round(validation_accuracy, 3)}")
    print(f"Test accuracy: {round(test_accuracy, 3)}")

    print_tree(model.result)


if __name__ == "__main__":
    full_test()
