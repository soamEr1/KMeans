import random

from src.kmeans import KMeans
from src.kmeans.kmeans import distance


CENTERS = [(0, 0), (10, 10), (-10, 10)]


def make_data(n, centers, seed):
    rng = random.Random(seed)
    X = []
    for i in range(n):
        cx, cy = centers[i % len(centers)]
        X.append([rng.gauss(cx, 0.5), rng.gauss(cy, 0.5)])
    return X


def test_assign_picks_nearest_centroid():
    X = [[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]]
    centroids = [[0, 0], [5, 5]]
    model = KMeans(n_clusters=2)
    assert model._assign(X, centroids) == [0, 0, 1, 1]


def test_fit_finds_known_centers():
    X = make_data(300, CENTERS, seed=0)
    model = KMeans(n_clusters=3, seed=42).fit(X)
    assert len(model.centroids_) == 3
    assert sorted(set(model.labels_)) == [0, 1, 2]
    for cx, cy in CENTERS:
        assert min(distance([cx, cy], c) for c in model.centroids_) < 1.0


def test_same_seed_same_result():
    X = make_data(300, CENTERS, seed=1)
    a = KMeans(n_clusters=3, seed=7).fit(X)
    b = KMeans(n_clusters=3, seed=7).fit(X)
    assert a.labels_ == b.labels_
    assert a.centroids_ == b.centroids_


def test_inertia_non_increasing():
    X = make_data(300, CENTERS, seed=2)
    values = []
    for it in range(1, 8):
        m = KMeans(n_clusters=3, max_iter=it, seed=3).fit(X)
        values.append(m.inertia(X, m.labels_))
    for prev, cur in zip(values, values[1:]):
        assert cur <= prev + 1e-9


def test_different_k_gives_different_cluster_count():
    X = make_data(300, CENTERS, seed=4)
    m2 = KMeans(n_clusters=2, seed=42).fit(X)
    m3 = KMeans(n_clusters=3, seed=42).fit(X)
    assert len(m2.centroids_) == 2
    assert len(m3.centroids_) == 3
