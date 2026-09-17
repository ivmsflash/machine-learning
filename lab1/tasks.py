"""
Лабораторная работа: Scikit-Learn
Автор: Власов М.С., группа 1-41м

Запуск:
  python tasks.py 2a     # линейная регрессия
  python tasks.py 2b     # классификация Iris
  python tasks.py 2c     # понижение размерности Iris (PCA)
  python tasks.py 2d     # кластеризация Iris (KMeans)
  python tasks.py 5      # распознавание рукописных цифр
  python tasks.py all    # всё подряд


Требуемые библиотеки:
  pip install numpy pandas matplotlib seaborn scikit-learn
"""

import sys
import matplotlib
matplotlib.use("TkAgg") ## чтобы открывались окна с графиками

# /////
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import make_regression, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score, accuracy_score, classification_report,
                             confusion_matrix, silhouette_score)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (8, 6)
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# П.2a. Линейная регрессия
# ---------------------------------------------------------------------
def task_2a_regression():
    print("\n=== П.2a. Линейная регрессия ===")

    # 1a. Синтетические данные
    X, y, coef_true = make_regression(n_samples=200, n_features=1,
                                      noise=15, bias=20, coef=True,
                                      random_state=RANDOM_STATE)
    df = pd.DataFrame({"x": X.ravel(), "y": y})
    print(df.head())

    # Визуализация исходных данных
    sns.scatterplot(data=df, x="x", y="y", s=60, alpha=0.7,
                    color="steelblue", edgecolor="k")
    plt.title("Синтетические данные для регрессии")
    plt.show()

    # 1b, 1c. Класс модели и гиперпараметры
    model = LinearRegression(fit_intercept=True)

    # 1d. X и y для fit()
    X_feat = df[["x"]].values
    y_target = df["y"].values

    # 1e. Обучение
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feat, y_target, test_size=0.2, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    print("Коэффициент:", model.coef_[0])
    print("Свободный член:", model.intercept_)

    # 1f. Предсказание для новых данных
    X_new = np.array([[3.5], [7.2], [12.0]])
    print("Прогноз для новых точек:", model.predict(X_new))

    # Визуализация результатов
    y_pred = model.predict(X_te)
    sns.scatterplot(x=X_te.ravel(), y=y_te, label="тест",
                    color="orange", s=60, edgecolor="k")
    sns.scatterplot(x=X_tr.ravel(), y=y_tr, label="обучение",
                    color="steelblue", alpha=0.5, s=40)
    plt.plot(X_tr, model.predict(X_tr), color="red", lw=2)
    plt.title(f"Линейная регрессия, R² = {r2_score(y_te, y_pred):.3f}")
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------
# П.2b. Классификация Iris
# ---------------------------------------------------------------------
def task_2b_classification():
    print("\n=== П.2b. Классификация Iris ===")

    # 1a. Загрузка Iris через Seaborn
    iris = sns.load_dataset("iris")
    print(iris.head())

    # Парные графики (п.3)
    sns.pairplot(iris, hue="species", palette="husl")
    plt.suptitle("Iris: парные графики", y=1.02)
    plt.show()

    # 1b, 1c. Модель и гиперпараметры
    model = LogisticRegression(C=1.0, max_iter=200,
                               random_state=RANDOM_STATE)

    # 1d. X и y
    features = ["sepal_length", "sepal_width",
                "petal_length", "petal_width"]
    X = iris[features].values
    y = iris["species"].values
    X = StandardScaler().fit_transform(X)

    # 1e. Обучение
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    model.fit(X_tr, y_tr)

    # 1f. Предсказание
    y_pred = model.predict(X_te)
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred))

    X_new = np.array([[5.1, 3.5, 1.4, 0.2],
                      [6.7, 3.0, 5.2, 2.3]])
    print("Прогноз для новых цветков:", model.predict(X_new))

    # Визуализация результатов — матрица ошибок
    cm = confusion_matrix(y_te, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion matrix")
    plt.xlabel("Предсказано"); plt.ylabel("Истинно")
    plt.show()


# ---------------------------------------------------------------------
# П.2c. Понижение размерности Iris (PCA)
# ---------------------------------------------------------------------
def task_2c_pca():
    print("\n=== П.2c. Понижение размерности Iris (PCA) ===")

    # 1a. Данные
    iris = sns.load_dataset("iris")
    sns.pairplot(iris, hue="species", palette="husl")
    plt.suptitle("Исходные данные Iris", y=1.02)
    plt.show()

    # 1b, 1c. Модель и гиперпараметры
    pca = PCA(n_components=2)

    # 1d. X
    features = ["sepal_length", "sepal_width",
                "petal_length", "petal_width"]
    X = StandardScaler().fit_transform(iris[features].values)

    # 1e. Обучение
    X_pca = pca.fit_transform(X)
    print("Объяснённая дисперсия:", pca.explained_variance_ratio_)

    # 1f. Новые данные
    X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
    print("Новая точка в PC:", pca.transform(X_new))

    # Визуализация результата
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["species"] = iris["species"].values
    sns.scatterplot(data=pca_df, x="PC1", y="PC2",
                    hue="species", palette="husl", s=70)
    plt.title("PCA: Iris в 2 компонентах")
    plt.show()


# ---------------------------------------------------------------------
# П.2d. Кластеризация Iris (KMeans)
# ---------------------------------------------------------------------
def task_2d_clustering():
    print("\n=== П.2d. Кластеризация Iris (KMeans) ===")

    # 1a. Данные
    iris = sns.load_dataset("iris")
    sns.pairplot(iris, hue="species", palette="husl")
    plt.suptitle("Исходные данные Iris", y=1.02)
    plt.show()

    features = ["sepal_length", "sepal_width",
                "petal_length", "petal_width"]
    X = StandardScaler().fit_transform(iris[features].values)

    # 1c. Подбор числа кластеров (метод локтя)
    inertia = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=RANDOM_STATE).fit(X)
        inertia.append(km.inertia_)
    sns.lineplot(x=list(range(2, 11)), y=inertia, marker="o")
    plt.xlabel("k"); plt.ylabel("Inertia")
    plt.title("Метод локтя")
    plt.show()

    # 1b, 1c. Модель с выбранными параметрами
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)

    # 1e. Обучение
    kmeans.fit(X)
    labels = kmeans.predict(X)
    print("Silhouette:", silhouette_score(X, labels))

    # 1f. Новые точки
    X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
    print("Кластер новой точки:", kmeans.predict(X_new))

    # Визуализация — через PCA
    X_vis = PCA(n_components=2).fit_transform(X)
    vis = pd.DataFrame(X_vis, columns=["PC1", "PC2"])
    vis["cluster"] = labels.astype(str)
    sns.scatterplot(data=vis, x="PC1", y="PC2",
                    hue="cluster", palette="viridis", s=70)
    plt.title("KMeans на Iris (k=3)")
    plt.show()


# ---------------------------------------------------------------------
# П.5. Распознавание рукописных цифр
# ---------------------------------------------------------------------
def task_5_digits():
    print("\n=== П.5. Распознавание рукописных цифр ===")

    # 5a. Загрузка и визуализация
    digits = load_digits()
    print("Размер:", digits.data.shape)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, img, lbl in zip(axes.ravel(), digits.images, digits.target):
        ax.imshow(img, cmap="gray_r")
        ax.set_title(lbl); ax.axis("off")
    plt.suptitle("Примеры цифр")
    plt.show()

    # 5b. Понижение размерности 64 → 2 (PCA, без учителя)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(digits.data)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["digit"] = digits.target.astype(str)
    sns.scatterplot(data=pca_df, x="PC1", y="PC2",
                    hue="digit", palette="tab10", s=30)
    plt.title("PCA: 64 → 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.show()

    # 5c. Классификация
    X_tr, X_te, y_tr, y_te = train_test_split(
        digits.data, digits.target, test_size=0.2,
        random_state=RANDOM_STATE, stratify=digits.target)
    model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # 5d. Оценка точности
    acc = accuracy_score(y_te, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_te, y_pred))

    # 5e. Матрица различий
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion matrix, accuracy = {acc:.4f}")
    plt.xlabel("Предсказано"); plt.ylabel("Истинно")
    plt.show()


# ---------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------
TASKS = {
    "2a": task_2a_regression,
    "2b": task_2b_classification,
    "2c": task_2c_pca,
    "2d": task_2d_clustering,
    "5":  task_5_digits,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python tasks.py [2a|2b|2c|2d|5|all]")
    elif sys.argv[1] == "all":
        for name, fn in TASKS.items():
            print(f"\n>>> {name}")
            fn()
    elif sys.argv[1] in TASKS:
        TASKS[sys.argv[1]]()
    else:
        print(f"Неизвестное задание: {sys.argv[1]}")