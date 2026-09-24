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

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# П.1. Отложенная выборка (holdout)
# ---------------------------------------------------------------------
def task_1():
    """
    Проверка модели классификации Iris на отложенной выборке.

    Идея метода (Плас, с. 410):
      данные делятся на две части — обучающую и проверочную.
      Модель учится на первой, оценивается на второй.
      Точность на проверочной части показывает, как модель
      будет работать на новых данных.
    """
    print("\n=== П.1. Отложенная выборка (holdout) ===")

    # load_iris() — загрузка встроенного набора данных Iris.
    # Параметры: нет.
    # Возвращает объект Bunch с полями:
    #   data         : ndarray (150, 4) — матрица признаков
    #   target       : ndarray (150,)   — метки классов (0, 1, 2)
    #   feature_names: list — имена признаков
    #   target_names : list — имена классов
    iris = load_iris()

    # X — матрица признаков формы (n_samples, n_features) = (150, 4)
    # y — вектор меток формы (n_samples,) = (150,)
    X = iris.data
    y = iris.target

    # train_test_split — делит данные на обучающую и проверочную части.
    # Параметры:
    #   *arrays      : массивы для разбиения (здесь X и y)
    #   test_size    : float — доля объектов в проверочной части
    #                  (0.5 = 50%)
    #   random_state : int — seed генератора случайных чисел,
    #                  чтобы разбиение было воспроизводимым
    # Возвращает: X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=RANDOM_STATE)

    # LogisticRegression — линейный классификатор.
    # Параметры:
    #   max_iter     : int — максимум итераций солвера
    #   random_state : int — seed
    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)

    # fit(X, y) — обучение модели на обучающей выборке.
    # Параметры:
    #   X : ndarray (n_samples, n_features) — признаки
    #   y : ndarray (n_samples,)            — метки классов
    # Возвращает: сам объект модели (self)
    model.fit(X_train, y_train)

    # predict(X) — предсказание меток классов для новых объектов.
    # accuracy_score(y_true, y_pred) — доля правильных ответов.
    # Считаем точность отдельно на обучении и на тесте.
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))

    print("Accuracy на обучении:", round(acc_train, 4))
    print("Accuracy на тесте   :", round(acc_test, 4))

    # Если точность на обучении сильно выше — модель переобучена.
    # Если обе низкие — модель недообучена.
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
    Это устойчивее, чем один holdout: все данные поочерёдно
    используются и для обучения, и для проверки.
    """
    print("\n=== П.2. Кросс-валидация ===")

    # Данные
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Модель, которую будем проверять
    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)

    # cross_val_score — оценка модели методом кросс-валидации.
    # Параметры:
    #   estimator : модель с методами fit/predict
    #   X, y      : данные
    #   cv        : int | KFold | LeaveOneOut | ... — схема разбиения
    #   scoring   : str — метрика ("accuracy" по умолчанию у классификатора)
    # Возвращает: ndarray с оценками по каждому фолду.

    # 5-fold (по умолчанию cv=5)
    scores_5 = cross_val_score(model, X, y, cv=5)
    print("5-fold:", np.round(scores_5, 3))
    print("  среднее:", round(scores_5.mean(), 4))

    # 2-fold — быстро, но грубо
    scores_2 = cross_val_score(model, X, y, cv=2)
    print("2-fold:", np.round(scores_2, 3))
    print("  среднее:", round(scores_2.mean(), 4))

    # LeaveOneOut — каждый объект по очереди становится тестом.
    # На Iris (150 объектов) это 150 обучений — медленно, но точно.
    scores_loo = cross_val_score(model, X, y, cv=LeaveOneOut())
    print("leave-one-out среднее:", round(scores_loo.mean(), 4))

    # barplot — столбики со средними оценками по трём схемам.
    # Параметры sns.barplot:
    #   x : list — подписи по оси X
    #   y : list — значения по оси Y
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

    # Сгенерируем данные по синусоиде с шумом
    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.rand(100, 1) * 6 - 3
    y = np.sin(X).ravel() + 0.3 * rng.randn(100)

    # Степени полинома, которые будем проверять
    degrees = range(1, 15)

    # validation_curve обучает модель для каждого значения степени
    train_scores, test_scores = validation_curve(
        make_pipeline(PolynomialFeatures(), LinearRegression()),
        X, y,
        param_name="polynomialfeatures__degree",
        param_range=degrees,
        cv=5)

    # Нарисуем средние значения
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

    # Найдём лучшую степень
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

    degree = 3   # фиксируем сложность

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

    # Что будем перебирать
    param_grid = {"polynomialfeatures__degree": range(1, 15)}

    # Ищем лучшую степень
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
        # Берём куски данных
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Обучаем модель на train
        model.fit(X_train, y_train)

        # Считаем accuracy на test
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

    # Моя функция
    my_scores = my_cross_val_score(model, X, y, cv=5)

    # Стандартная sklearn
    sk_scores = cross_val_score(model, X, y, cv=5)

    print("Моя:     ", np.round(my_scores, 3))
    print("sklearn: ", np.round(sk_scores, 3))
    print("Среднее моё:     ", round(my_scores.mean(), 4))
    print("Среднее sklearn: ", round(sk_scores.mean(), 4))


# ////////////
# Запуск
# ////////////
TASKS = {
    "1": task_1,
    "2": task_2,
    "3": task_3,
    "4": task_4,
    "5": task_5,
    "7": task_7,
    "8": task_8,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python tasks.py [1|2|3|4|5|7|8|all]")
    elif sys.argv[1] == "all":
        for name, fn in TASKS.items():
            print(f"\n>>> {name}")
            fn()
    elif sys.argv[1] in TASKS:
        TASKS[sys.argv[1]]()
    else:
        print("Неизвестный пункт:", sys.argv[1])