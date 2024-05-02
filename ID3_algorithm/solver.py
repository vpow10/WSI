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
        # self.data_train, self.data_test = train_test_split(data, test_size=0.2)
        # self.data_test, self.data_val = train_test_split(self.data_test, test_size=0.5)
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
                mean = round(X_col_temp.mean())
                X[col] = X[col].fillna(mean)
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
        predictions = []
        for _, sample in X.iterrows():
            node = tree
            while node.children:
                node = node.children[sample[node.feature]]
            predictions.append(node.result)
        return predictions

    def accuracy(self, Y_true, Y_pred):
        return sum([Y_true[i] == Y_pred[i] for i in range(len(Y_true))]) / len(Y_true)


# df = pd.read_csv('data/cardio_train.csv', sep=';', index_col='id')
# print(df['ap_hi'].max())
# id3 = ID3_algorithm(df)

# data = np.array([40000])
# data = pd.cut(data, bins=[0, 35*365.25, 45*365.25, 65*365.25], labels=[0, 1, 2])
# if math.isnan(data[0]):
#     print('yes')

# df = pd.read_csv('data/test.csv', sep=';')
# df['x2'] = pd.cut(df['x2'], bins=[0, 1, 2], labels=[1, 2])
# X = df.iloc[:, :-1]
# Y = df.iloc[:, -1]
# id3 = ID3_algorithm(df)
# # print(id3.information_gain(X, Y, 'x1'))
# # print(id3.information_gain(X, Y, 'x2'))


# # def print_tree(node, depth=0):
# #     if node.result is not None:
# #         print("  " * depth, "Result:", node.result)
# #     else:
# #         print("  " * depth, "Feature:", node.feature)
# #         for value, child in node.children.items():
# #             print("  " * (depth+1), f"Value {value}:")
# #             print_tree(child, depth + 2)

# root = id3.fit(X, Y, 9)
# predictions = id3.predict(X, root)
# print(predictions)
# print(id3.accuracy(Y.tolist(), predictions))
# print_tree(root)