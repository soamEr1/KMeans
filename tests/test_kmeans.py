import numpy as np
from sklearn.datasets import make_blobs
from src.kmeans import KMeans


def test_assign_picks_nearest_centroid():
    X = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]])
    centroids = np.array([[0, 0], [5, 5]])
    model = KMeans(k=2)

    labels = model._assign(X, centroids)
    assert labels.tolist() == [0, 0, 1, 1]


def test_fit_returns_three_centroids():
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)

    model = KMeans(k=3).fit(X)

    assert model.centroids_.shape == (3, 2)
    assert len(np.unique(model.labels_)) == 3