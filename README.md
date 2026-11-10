# KMeans

Кластеризация k-средних на чистом Python (без numpy в ядре алгоритма). Для визуализации используется matplotlib.

## Структура

- `src/kmeans/kmeans.py` — класс `KMeans`
- `src/kmeans/__init__.py` — экспорт класса
- `kmeans.py` — CLI: генерация данных, кластеризация, график, метод локтя
- `tests/test_kmeans.py` — тесты

## Запуск

Команды выполняются из корня проекта:

python3 kmeans.py --samples 300 --k 3 --iter 100 --seed 42
python3 kmeans.py --plot
python3 kmeans.py --elbow

Аргументы:
- `--samples` — количество точек (по умолчанию 300)
- `--k` — количество кластеров (по умолчанию 3)
- `--iter` — максимум итераций (по умолчанию 100)
- `--seed` — зерно генератора случайных чисел (по умолчанию 42)
- `--plot` — показать график кластеров
- `--elbow` — показать график метода локтя

## Тесты

python3 -m pytest tests/

## Использование в коде

from src.kmeans import KMeans

model = KMeans(n_clusters=3, max_iter=100, seed=42).fit(X)
labels = model.labels_
centroids = model.centroids_
inertia = model.inertia(X, labels)
