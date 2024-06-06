import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin


class Solver(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_priors = {}
        self.feature_stats = {}
        self.classes = None
        self.classes_ = None # for sklearn compatibility

    def calculate_classes_prob(self, y):
        classes, counts = np.unique(y, return_counts=True)
        class_prob = counts / y.shape[0]
        self.class_priors = dict(zip(classes, class_prob))
        self.classes = classes
        self.classes_ = classes  # Added for scikit-learn compatibility

    def calculate_condition_prob(self, X, y):
        self.feature_stats = {}
        for c in self.classes:
            class_features = X[y == c]
            self.feature_stats[c] = {
                'mean': np.mean(class_features, axis=0),
                'std': np.std(class_features, axis=0)
            }

    def fit(self, X, y):
        self.calculate_classes_prob(y)
        self.calculate_condition_prob(X, y)
        # to match sklearn cross valdation
        return self

    def calculate_posterior(self, x):
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.class_priors[c])
            conditional = np.sum(np.log(self._calculate_likelihood(x, c)))
            posteriors[c] = prior + conditional
        return posteriors

    def _calculate_likelihood(self, x, c):
        mean = self.feature_stats[c]['mean']
        std = self.feature_stats[c]['std']
        likelihood = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return likelihood

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = self.calculate_posterior(x)
            y_pred.append(max(posteriors, key=posteriors.get))
        return np.array(y_pred)
