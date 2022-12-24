import numpy as np

class MyLogisticRegression():
    
    def __init__(self, lr=0.1, n_iters=4000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            pred = np.dot(X, self.weights) + self.bias
            pred = self.sigmoid(pred)

            dw = (1/n_samples) * np.dot(X.T, (pred - y))
            db = (1/n_samples) * np.sum(pred-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(y_pred)
        return y_pred

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))