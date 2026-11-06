import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

        self.random_state = random_state

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=self.k, replace=False)
        return X[idx].copy()

    def _assign(self, X, centroids):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

        return np.argmin(distances, axis=1)

    def _update(self, X, labels, centroids):
        new_centroids = np.zeros_like(centroids)
        for j in range(self.k):

            points = X[labels == j]
            if len(points) > 0:
                new_centroids[j] = points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        return new_centroids

    def fit(self, X):
        centroids = self._init_centroids(X)
        for _ in range(self.max_iter):

            labels = self._assign(X, centroids)
            new_centroids = self._update(X, labels, centroids)
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:

                break
        self.centroids_ = centroids

        self.labels_ = self._assign(X, centroids)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)