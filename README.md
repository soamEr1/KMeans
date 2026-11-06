# KMeans

Реализация алгоритма k-средних на numpy.

## Структура

- `src/kmeans/kmeans.py` — класс `KMeans`
- `tests/test_kmeans.py` — тесты

## Запуск тестов

python3 -m pytest tests/

## Использование

from src.kmeans import KMeans

model = KMeans(k=3).fit(X)
labels = model.labels_
centroids = model.centroids_
