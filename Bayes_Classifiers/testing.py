from solver import Solver
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


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


def split_test(first_split: float, second_split: float):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=first_split, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=second_split, random_state=42)
    solver = Solver()
    solver.fit(X_train, y_train)
    y_pred_val = solver.predict(X_val)
    y_pred_test = solver.predict(X_test)
    print(f'Accuracy on validation set: {accuracy_score(y_val, y_pred_val)}')
    print(f'Accuracy on test set: {accuracy_score(y_test, y_pred_test)}')

def cross_val(cv: int):
    X, y = load_breast_cancer(return_X_y=True)
    solver = Solver()
    scores = cross_val_score(solver, X, y, cv=cv, scoring='accuracy')
    print(f'Cross-validation scores: {scores}')
    print(f'Mean cross-validation score: {np.mean(scores)}')


if __name__=='__main__':
    split_test(0.1, 0.5)
    cross_val(5)