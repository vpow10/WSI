import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prep_data():
    digits = load_digits()
    X_mnist, y_mnist = digits.data, digits.target
    X_mnist_train, X_mnist_val, y_mnist_train, y_mnist_val = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)
    X_mnist_val, X_mnist_test, y_mnist_val, y_mnist_test = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)
    scaler_mnist = StandardScaler()
    X_mnist_train_scaled = scaler_mnist.fit_transform(X_mnist_train)
    X_mnist_val_scaled = scaler_mnist.transform(X_mnist_val)
    X_mnist_test_scaled = scaler_mnist.transform(X_mnist_test)
    return X_mnist_train_scaled, y_mnist_train, X_mnist_val_scaled, y_mnist_val, X_mnist_test_scaled, y_mnist_test

