import numpy as np

class KMeans:
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_centroids(self, X):
        pass

    def _assign(self, X, centroids):
        pass

    def _update(self, X, labels):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass
