import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from solver import ID3_algorithm, Node
from data_prep import discretize


def single_prediction(depth: int):
    data = pd.read_csv('data/cardio_train.csv', sep=';', index_col="id")
    data = discretize(data)
    data_train, data_test = train_test_split(data, test_size=0.2)
    data_test, data_val = train_test_split(data_test, test_size=0.5)
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
    print(f"Training accuracy: {model.accuracy(Y_train.tolist(), model.predictions)}")

    # Validation accuracy

    validation_predictions = model.predict(X_val, model.result)
    print(f"Validation accuracy: {model.accuracy(Y_val.tolist(), validation_predictions)}")

    # Test accuracy

    test_predictions = model.predict(X_test, model.result)
    print(f"Test accuracy: {model.accuracy(Y_test.tolist(), test_predictions)}")


single_prediction(999)