import argparse
import random

import matplotlib.pyplot as plt

from src.kmeans import KMeans


def make_samples(n, k, seed):
    rng = random.Random(seed)
    centers = [(rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(k)]
    X = []
    for i in range(n):
        cx, cy = centers[i % k]
        X.append([rng.gauss(cx, 1.0), rng.gauss(cy, 1.0)])
    return X


def plot_elbow(X, max_k, iters, seed):
    ks = list(range(1, max_k + 1))
    values = []
    for k in ks:
        m = KMeans(n_clusters=k, max_iter=iters, seed=seed).fit(X)
        values.append(m.inertia(X, m.labels_))
    plt.figure(figsize=(6, 4))
    plt.plot(ks, values, "o-")
    plt.xlabel("k")
    plt.ylabel("inertia")
    plt.title("Метод локтя")
    plt.grid(True)
    plt.show()


def plot_clusters(X, model):
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c=model.labels_, cmap="viridis", s=15)
    plt.scatter(
        [c[0] for c in model.centroids_],
        [c[1] for c in model.centroids_],
        c="red", marker="X", s=200, label="центры",
    )
    plt.legend()
    plt.title("Кластеры")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elbow", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    X = make_samples(args.samples, args.k, args.seed)

    if args.elbow:
        plot_elbow(X, max_k=8, iters=args.iter, seed=args.seed)
        return

    model = KMeans(n_clusters=args.k, max_iter=args.iter, seed=args.seed).fit(X)
    sizes = [model.labels_.count(j) for j in range(args.k)]
    print("Размеры кластеров:", sizes)
    print(f"Inertia: {model.inertia(X, model.labels_):.2f}")

    if args.plot:
        plot_clusters(X, model)


if __name__ == "__main__":
    main()
