import numpy as np
from src.kmeans import KMeans

def test_assign_picks_nearest_centroid():
    X = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]])
    centroids = np.array([[0, 0], [5, 5]])
    model = KMeans(k=2)

    labels = model._assign(X, centroids)
    assert labels.tolist() == [0, 0, 1, 1]
