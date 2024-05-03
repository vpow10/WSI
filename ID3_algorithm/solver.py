import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import math


class Node:
    def __init__(self, feature=None, children=None, result=None):
        self.feature = feature  # Splitting feature
        self.children = children  # Dictionary to store children nodes
        self.result = result  # Result if leaf node


class ID3_algorithm():
    def __init__(self, data: pd.core.frame.DataFrame):
        self.data = data
        self.result = None
        self.predictions = None

    def entropy(self, Y: pd.core.frame.DataFrame):
        _, counts = np.unique(Y, return_counts=True)
        probabilities = counts / len(Y)
        return -np.sum(probabilities * np.log(probabilities))

    def information_gain(self, X: pd.core.frame.DataFrame, Y: pd.core.frame.DataFrame, feature: str):
        entropy_before = self.entropy(Y)
        unique_values = np.unique(X[feature])
        entropy_after = 0
        for value in unique_values:
            Y_subset = Y[X[feature] == value]
            data_subset = X[X[feature] == value]
            entropy_after += len(data_subset) / len(X) * self.entropy(Y_subset)
        return entropy_before - entropy_after

    def get_parameters(self):
        pass

    def fit(self, X, Y, depth: int):
        # Check for NaN values and replace them with the mean of the column
        for col in X.columns:
            if X[col].isnull().values.any():
                X_col_temp = X[col]
                X_col_temp.dropna(inplace=True)
                X_col_temp = X_col_temp.astype(float)
                try:
                    mean = round(X_col_temp.mean())
                    X[col] = X[col].fillna(mean).astype(float)
                except ValueError:      # if every value in a column is NaN
                    raise ValueError("All values in a column are NaN")
        if len(set(Y)) == 1:
            # If all the labels are the same, return a leaf node
            return Node(result=Y.iloc[0])
        if len(X.keys()) == 0 or depth == 0:
            # If no features are left or depth is 0, return a leaf node
            counts = Y.value_counts()
            return Node(result=counts.idxmax())
        best_feature = None
        best_gain = -1
        for feature in X.keys():
            gain = self.information_gain(X, Y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        if best_gain == 0:
            # If no gain is achieved, return a leaf node
            counts = Y.value_counts()
            return Node(result=counts.idxmax())

        node = Node(feature=best_feature)
        unique_values = np.unique(X[best_feature])
        node.children = {}
        for value in unique_values:
            Y_subset = Y[X[best_feature] == value]
            X_subset = X[X[best_feature] == value]
            node.children[value] = self.fit(X_subset, Y_subset, depth-1)

        return node

    def predict(self, X, tree):
        # Check for NaN values and replace them with the mean of the column
        for col in X.columns:
            if X[col].isnull().values.any():
                X_col_temp = X[col]
                X_col_temp.dropna(inplace=True)
                X_col_temp = X_col_temp.astype(float)
                try:   # if every value in a column is NaN
                    mean = round(X_col_temp.mean())
                    X[col] = X[col].fillna(mean)
                except ValueError:
                    raise ValueError("All values in a column are NaN")
        predictions = []
        for _, sample in X.iterrows():
            node = tree
            while node.children:
                if sample[node.feature] not in node.children:
                    break
                node = node.children[sample[node.feature]]
            predictions.append(node.result)
        return predictions

    def accuracy(self, Y_true, Y_pred):
        return sum([Y_true[i] == Y_pred[i] for i in range(len(Y_true))]) / len(Y_true)
