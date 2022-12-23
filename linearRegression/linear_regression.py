import numpy as np
import matplotlib.pyplot as plt
from json import load

class MyLinearRegression():
    def __init__(self, lr=0.0001, iters=3000):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)
        self.weights = 0
        self.bias = 0

        for _ in range(self.iters):
            y_pred = np.dot(self.weights, X) + self.bias

            dw = (-2 / n_samples) * np.dot(X, (y - y_pred))
            db = (-2 / n_samples) * np.sum(y-y_pred)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        X = np.array(X)
        y_pred = np.dot(X.T, self.weights) + self.bias
        return y_pred