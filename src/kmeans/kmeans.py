import math
import random


def distance(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _init_centroids(self, X):
        rng = random.Random(self.seed)
        return [list(p) for p in rng.sample(X, self.n_clusters)]

    def _assign(self, X, centroids):
        labels = []
        for p in X:
            dists = [distance(p, c) for c in centroids]
            labels.append(dists.index(min(dists)))
        return labels

    def _update(self, X, labels, centroids):
        new_centroids = []
        for j in range(self.n_clusters):
            points = [p for p, lab in zip(X, labels) if lab == j]
            if points:
                dim = len(points[0])
                new_centroids.append([sum(p[d] for p in points) / len(points) for d in range(dim)])
            else:
                new_centroids.append(list(centroids[j]))
        return new_centroids

    def fit(self, X):
        centroids = self._init_centroids(X)
        for _ in range(self.max_iter):
            labels = self._assign(X, centroids)
            new_centroids = self._update(X, labels, centroids)
            shift = max(distance(a, b) for a, b in zip(new_centroids, centroids))
            centroids = new_centroids
            if shift < self.tol:
                break
        self.centroids_ = centroids
        self.labels_ = self._assign(X, centroids)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)

    def inertia(self, X, labels):
        return sum(distance(p, self.centroids_[lab]) ** 2 for p, lab in zip(X, labels))
