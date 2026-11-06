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

    def _update(self, X, labels):
        pass
    def fit(self, X):
        pass

    def predict(self, X):
        pass