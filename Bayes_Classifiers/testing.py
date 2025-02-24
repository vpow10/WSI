from solver import Solver
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


def simple_test():
    data = np.array([
    [5.1, 3.5, 0],
    [4.9, 3.0, 0],
    [4.7, 3.2, 0],
    [4.6, 3.1, 0],
    [5.0, 3.6, 0],
    [5.4, 3.9, 0],
    [4.6, 3.4, 0],
    [5.0, 3.4, 0],
    [4.4, 2.9, 0],
    [4.9, 3.1, 0],
    [5.4, 3.7, 0],
    [4.8, 3.4, 0],
    [4.8, 3.0, 0],
    [4.3, 3.0, 0],
    [5.8, 4.0, 0],
    [5.7, 4.4, 1],
    [5.4, 3.9, 1],
    [5.1, 3.5, 1],
    [5.7, 3.8, 1],
    [5.1, 3.8, 1],
    [5.4, 3.4, 1],
    [5.1, 3.7, 1],
    [4.6, 3.6, 1],
    [5.1, 3.3, 1],
    [5.4, 3.4, 1],
    [4.8, 3.4, 1],
    [5.0, 3.0, 1],
    [5.0, 3.4, 1],
    [5.2, 3.5, 1],
    [5.2, 3.4, 1]
    ])

    # Splitting into features and target
    X = data[:, :2]
    y = data[:, 2]

    # Using sklearn GaussianNB for comparison
    clf = GaussianNB()
    clf.fit(X, y)
    print("Sklearn GaussianNB Predictions:", clf.predict(X))
    print("Sklearn GaussianNB Accuracy:", accuracy_score(y, clf.predict(X)) * 100, '%')

    # Using the custom Solver class
    solver = Solver()
    solver.fit(X, y)
    print("Custom Solver Predictions:", solver.predict(X))
    print("Custom Solver Accuracy:", accuracy_score(y, solver.predict(X)) * 100, '%')


def split_test(first_split: float):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=first_split, random_state=42)
    solver = Solver()
    solver.fit(X_train, y_train)
    y_pred_val = solver.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    std = np.std(y_val == y_pred_val)
    print(f'Accuracy on validation set: {accuracy}')
    return accuracy, std

def cross_val(cv: int):
    X, y = load_breast_cancer(return_X_y=True)
    solver = Solver()
    scores = cross_val_score(solver, X, y, cv=cv, scoring='accuracy')
    print(f'Cross-validation scores: {scores}')
    print(f'Mean cross-validation score: {np.mean(scores)}')
    std = np.std(scores)
    return np.mean(scores), std

def splitting_results():
    accuracies = []
    stds = []
    for i in range(1, 10):
        accuracy, std = split_test(i/10)
        accuracies.append(accuracy)
        stds.append(std)
    plt.figure(1)
    plt.plot(np.arange(1, 10)*10, accuracies)
    plt.xlabel('Validation set size (%)')
    plt.ylabel('Accuracy')
    plt.title('Validation set size vs Accuracy')
    plt.figure(2)
    plt.plot(np.arange(1, 10)*10, stds)
    plt.xlabel('Validation set size (%)')
    plt.ylabel('Standard deviation')
    plt.title('Validation set size vs Standard deviation')
    plt.show()

def cross_val_results():
    accuracies = []
    stds = []
    for i in range(2, 11):
        accuracy, std = cross_val(i)
        accuracies.append(accuracy)
        stds.append(std)
    plt.figure(1)
    plt.plot(np.arange(2, 11), accuracies)
    plt.xlabel('Number of folds')
    plt.ylabel('Accuracy')
    plt.title('Number of folds vs Accuracy')
    plt.figure(2)
    plt.plot(np.arange(2, 11), stds)
    plt.xlabel('Number of folds')
    plt.ylabel('Standard deviation')
    plt.title('Number of folds vs Standard deviation')
    plt.show()

def final_test():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
    solver = Solver()
    solver.fit(X_train, y_train)
    y_pred_val = solver.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f'Accuracy on validation set with splitting: {accuracy}')
    y_pred_test = solver.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f'Accuracy on test set with splitting: {accuracy}')
    cross_val(10)


if __name__=='__main__':
    simple_test()
    splitting_results()
    cross_val_results()
    final_test()