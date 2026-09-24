"""
Лабораторная работа: Scikit-Learn, гиперпараметры и проверка модели
Автор: Власов М.С., группа 1-41м
Преподаватель: Кокин Владимир Модестович

//////

Требуемые библиотеки:
    pip install -r requirements.txt
или вручную:
    pip install numpy matplotlib seaborn scikit-learn

//////

Запуск:
    python tasks.py 1        # отложенная выборка
    python tasks.py 2        # кросс-валидация
    python tasks.py 3        # кривые проверки
    python tasks.py 4        # кривые обучения
    python tasks.py 5        # поиск по сетке
    python tasks.py 7        # своя cross_val_score
    python tasks.py 8        # сравнение с sklearn
    python tasks.py 9        # свой PolynomialRegression
    python tasks.py all      # все задания подряд
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     KFold, LeaveOneOut,
                                     validation_curve, learning_curve,
                                     GridSearchCV)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, RegressorMixin

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# П.1. Отложенная выборка (holdout)
# ---------------------------------------------------------------------
def task_1():
    """
    Проверка модели классификации Iris на отложенной выборке.

    Идея метода (VanderPlas, с. 410):
      данные делятся на две части — обучающую и проверочную.
      Модель учится на первой, оценивается на второй.
    """
    print("\n=== П.1. Отложенная выборка (holdout) ===")

    # load_iris() — загрузка встроенного набора данных Iris.
    # Возвращает объект Bunch с полями:
    #   data   : ndarray (150, 4) — матрица признаков
    #   target : ndarray (150,)   — метки классов (0, 1, 2)
    iris = load_iris()

    # X — матрица признаков формы (n_samples, n_features) = (150, 4)
    # y — вектор меток формы (n_samples,) = (150,)
    X = iris.data
    y = iris.target

    # train_test_split — делит данные на обучающую и проверочную части.
    # Параметры:
    #   *arrays      : массивы для разбиения (здесь X и y)
    #   test_size    : float — доля объектов в проверочной части (0.5 = 50%)
    #   random_state : int — seed для воспроизводимости разбиения
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=RANDOM_STATE)

    # LogisticRegression — линейный классификатор.
    # Параметры:
    #   max_iter     : int — максимум итераций солвера
    #   random_state : int — seed
    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)

    # fit(X, y) — обучение модели.
    # Параметры:
    #   X : ndarray (n_samples, n_features) — признаки
    #   y : ndarray (n_samples,)            — метки классов
    model.fit(X_train, y_train)

    # accuracy_score — доля правильных ответов.
    # Считаем точность отдельно на обучении и на тесте.
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))

    print("Accuracy на обучении:", round(acc_train, 4))
    print("Accuracy на тесте   :", round(acc_test, 4))

    if acc_train - acc_test > 0.05:
        print("Модель переобучена")
    elif acc_test < 0.8:
        print("Модель недообучена")
    else:
        print("Модель нормальная")


# ---------------------------------------------------------------------
# П.2. Кросс-валидация
# ---------------------------------------------------------------------
def task_2():
    """
    Сравнение схем кросс-валидации на Iris (VanderPlas, с. 411).

    Кросс-валидация:
      данные делятся на k блоков (фолдов);
      модель обучается k раз: на k−1 фолдах, проверяется на оставшемся;
      итоговая оценка — среднее по фолдам.
    """
    print("\n=== П.2. Кросс-валидация ===")

    iris = load_iris()
    X = iris.data
    y = iris.target

    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)

    # cross_val_score — оценка модели методом кросс-валидации.
    # Параметры:
    #   estimator : модель с методами fit/predict
    #   X, y      : данные
    #   cv        : int | KFold | LeaveOneOut | ... — схема разбиения
    # Возвращает: ndarray с оценками по каждому фолду.

    scores_5 = cross_val_score(model, X, y, cv=5)
    print("5-fold:", np.round(scores_5, 3))
    print("  среднее:", round(scores_5.mean(), 4))

    scores_2 = cross_val_score(model, X, y, cv=2)
    print("2-fold:", np.round(scores_2, 3))
    print("  среднее:", round(scores_2.mean(), 4))

    scores_loo = cross_val_score(model, X, y, cv=LeaveOneOut())
    print("leave-one-out среднее:", round(scores_loo.mean(), 4))

    # sns.barplot — столбики со средними оценками
    sns.barplot(x=["2-fold", "5-fold", "LOO"],
                y=[scores_2.mean(), scores_5.mean(), scores_loo.mean()])
    plt.ylabel("Средняя accuracy")
    plt.title("Сравнение кросс-валидаций")
    plt.show()


# ---------------------------------------------------------------------
# П.3. Кривые проверки для полиномиальной регрессии
# ---------------------------------------------------------------------
def task_3():
    print("\n=== П.3. Кривые проверки ===")

    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.rand(100, 1) * 6 - 3
    y = np.sin(X).ravel() + 0.3 * rng.randn(100)

    degrees = range(1, 15)

    # validation_curve обучает модель для каждого значения степени
    train_scores, test_scores = validation_curve(
        make_pipeline(PolynomialFeatures(), LinearRegression()),
        X, y,
        param_name="polynomialfeatures__degree",
        param_range=degrees,
        cv=5)

    plt.plot(list(degrees), train_scores.mean(axis=1), marker="o",
             label="обучение")
    plt.plot(list(degrees), test_scores.mean(axis=1), marker="s",
             label="кросс-валидация")
    plt.xlabel("Степень полинома")
    plt.ylabel("R²")
    plt.title("Кривая проверки")
    plt.legend()
    plt.grid(True)
    plt.show()

    test_mean = test_scores.mean(axis=1)
    best = list(degrees)[int(np.argmax(test_mean))]
    print("Лучшая степень полинома:", best)


# ---------------------------------------------------------------------
# П.4. Кривые обучения
# ---------------------------------------------------------------------
def task_4():
    print("\n=== П.4. Кривые обучения ===")

    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.rand(100, 1) * 6 - 3
    y = np.sin(X).ravel() + 0.3 * rng.randn(100)

    degree = 3

    sizes, train_scores, test_scores = learning_curve(
        make_pipeline(PolynomialFeatures(degree), LinearRegression()),
        X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5)

    plt.plot(sizes, train_scores.mean(axis=1), marker="o", label="обучение")
    plt.plot(sizes, test_scores.mean(axis=1), marker="s",
             label="кросс-валидация")
    plt.xlabel("Размер обучающей выборки")
    plt.ylabel("R²")
    plt.title("Кривая обучения (степень = 3)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------
# П.5. Поиск по сетке
# ---------------------------------------------------------------------
def task_5():
    print("\n=== П.5. Поиск по сетке ===")

    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.rand(100, 1) * 6 - 3
    y = np.sin(X).ravel() + 0.3 * rng.randn(100)

    param_grid = {"polynomialfeatures__degree": range(1, 15)}

    grid = GridSearchCV(
        make_pipeline(PolynomialFeatures(), LinearRegression()),
        param_grid, cv=5)
    grid.fit(X, y)

    print("Лучшие параметры:", grid.best_params_)
    print("Лучшее качество:", round(grid.best_score_, 4))


# ---------------------------------------------------------------------
# П.7. Своя cross_val_score
# ---------------------------------------------------------------------
def my_cross_val_score(model, X, y, cv=5):
    """Своя простая кросс-валидация."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        scores.append(acc)

    return np.array(scores)


def task_7():
    print("\n=== П.7. Своя cross_val_score ===")

    iris = load_iris()
    X = iris.data
    y = iris.target

    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    scores = my_cross_val_score(model, X, y, cv=5)

    print("Мои оценки:", np.round(scores, 3))
    print("Среднее:", round(scores.mean(), 4))


# ---------------------------------------------------------------------
# П.8. Сравнение с sklearn
# ---------------------------------------------------------------------
def task_8():
    print("\n=== П.8. Сравнение с sklearn cross_val_score ===")

    iris = load_iris()
    X = iris.data
    y = iris.target

    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)

    my_scores = my_cross_val_score(model, X, y, cv=5)
    sk_scores = cross_val_score(model, X, y, cv=5)

    print("Моя:     ", np.round(my_scores, 3))
    print("sklearn: ", np.round(sk_scores, 3))
    print("Среднее моё:     ", round(my_scores.mean(), 4))
    print("Среднее sklearn: ", round(sk_scores.mean(), 4))


# ---------------------------------------------------------------------
# П.9*. Свой класс PolynomialRegression
# ---------------------------------------------------------------------
class PolynomialRegression(BaseEstimator, RegressorMixin):
    """
    Полиномиальная регрессия — аналог make_pipeline(
        PolynomialFeatures(degree), LinearRegression()).

    Наследуется от BaseEstimator и RegressorMixin, чтобы корректно
    работать с cross_val_score, GridSearchCV, clone и validation_curve.

    Параметры:
        degree        : int  — степень полинома
        fit_intercept : bool — вычислять ли свободный член в линейной регрессии
    """
    def __init__(self, degree=2, fit_intercept=True):
        self.degree = degree
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Обучает модель: строит полиномиальные признаки и обучает линейную регрессию."""
        self.poly_ = PolynomialFeatures(self.degree, include_bias=False)
        X_poly = self.poly_.fit_transform(X)

        self.lin_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.lin_.fit(X_poly, y)

        return self

    def predict(self, X):
        """Предсказывает значения."""
        return self.lin_.predict(self.poly_.transform(X))


def task_9():
    print("\n=== П.9*. Свой класс PolynomialRegression ===")

    # Данные как в п.3
    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.rand(100, 1) * 6 - 3
    y = np.sin(X).ravel() + 0.3 * rng.randn(100)

    # Свой класс
    poly_mine = PolynomialRegression(degree=3)
    poly_mine.fit(X, y)
    y_mine = poly_mine.predict(X)

    # Эквивалент через Pipeline
    poly_pipe = make_pipeline(PolynomialFeatures(3), LinearRegression())
    poly_pipe.fit(X, y)
    y_pipe = poly_pipe.predict(X)

    diff = float(np.max(np.abs(y_mine - y_pipe)))
    print("Максимальная разница предсказаний:", round(diff, 10))
    if diff < 1e-8:
        print("Классы эквивалентны")

    # Работа своего класса в cross_val_score
    scores = cross_val_score(PolynomialRegression(degree=3), X, y, cv=5)
    print("cross_val_score R²:", np.round(scores, 3),
          "| среднее:", round(scores.mean(), 4))

    # Работа своего класса в GridSearchCV
    grid = GridSearchCV(PolynomialRegression(), {"degree": range(1, 15)}, cv=5)
    grid.fit(X, y)
    print("GridSearchCV лучшая степень:", grid.best_params_)
    print("GridSearchCV лучшее R²:", round(grid.best_score_, 4))

    # График: свои предсказания против данных
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_plot = poly_mine.predict(X_plot)

    plt.scatter(X, y, s=30, alpha=0.6, label="данные")
    plt.plot(X_plot, y_plot, color="red", lw=2, label="PolynomialRegression(3)")
    plt.title("Свой класс PolynomialRegression")
    plt.legend()
    plt.grid(True)
    plt.show()


# ------
# Запуск
# ------
TASKS = {
    "1": task_1,
    "2": task_2,
    "3": task_3,
    "4": task_4,
    "5": task_5,
    "7": task_7,
    "8": task_8,
    "9": task_9,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python tasks.py [1|2|3|4|5|7|8|9|all]")
    elif sys.argv[1] == "all":
        for name, fn in TASKS.items():
            print(f"\n>>> {name}")
            fn()
    elif sys.argv[1] in TASKS:
        TASKS[sys.argv[1]]()
    else:
        print("Неизвестный пункт:", sys.argv[1])