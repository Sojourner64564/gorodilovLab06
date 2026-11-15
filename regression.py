# -*- coding: utf-8 -*-
"""
Лабораторная работа: регрессия, данные Boston Housing.
Файл с данными должен лежать рядом со скриптом (или поменяйте путь ниже).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

FILE = "boston.csv"   # для вашего файла: "boston___копия.csv"
RANDOM_STATE = 42


def metrics(model, X_train, X_test, y_train, y_test, name):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    res = {
        "Модель": name,
        "R2 train": r2_score(y_train, pred_train),
        "R2 test": r2_score(y_test, pred_test),
        "RMSE train": np.sqrt(mean_squared_error(y_train, pred_train)),
        "RMSE test": np.sqrt(mean_squared_error(y_test, pred_test)),
    }
    print(f"{name}: R2 train={res['R2 train']:.3f}, R2 test={res['R2 test']:.3f}, "
          f"RMSE train={res['RMSE train']:.3f}, RMSE test={res['RMSE test']:.3f}")
    return res


# 1) Загрузка
df = pd.read_csv(FILE)
print(df.head(), "\n")

# 2) Проверка типов данных
print("Типы данных:\n", df.dtypes, "\n")
non_numeric = df.select_dtypes(exclude="number").columns.tolist()
if non_numeric:
    df[non_numeric] = df[non_numeric].apply(pd.to_numeric, errors="coerce")
print("Все столбцы числовые:", not non_numeric, "\n")

# 3) Пропущенные значения
print("Пропуски по столбцам:\n", df.isna().sum(), "\n")
df = df.fillna(df.median())

# 4) Корреляции
corr = df.corr()
print("Корреляционная матрица:\n", corr.round(2), "\n")

# 5) Тепловая карта
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Корреляционная матрица (Boston)")
plt.tight_layout()
plt.show()

# 6) Выбор 4-6 признаков, лучше всего связанных с MEDV
corr_target = corr["MEDV"].drop("MEDV")
top = corr_target.abs().sort_values(ascending=False).head(5).index.tolist()
print("Топ-5 признаков по |корреляции| с MEDV:")
print(corr_target[top].round(3), "\n")

# 7) Диаграммы рассеяния признак–MEDV
fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 4))
for ax, col in zip(np.atleast_1d(axes), top):
    ax.scatter(df[col], df["MEDV"], alpha=0.6)
    ax.set_xlabel(col)
    ax.set_ylabel("MEDV")
    ax.set_title(f"{col} vs MEDV (r={corr.loc[col, 'MEDV']:.2f})")
plt.tight_layout()
plt.show()

# 8) Отбор по графикам
features = top

# 9) Факторные признаки и целевая
X = df[features]
y = df["MEDV"]

# 10) Разбиение 75:25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, shuffle=True)
print("Размеры выборок:", X_train.shape, X_test.shape, "\n")

# 11-13) Линейная регрессия и метрики
print("=== Основная часть ===")
lin = LinearRegression().fit(X_train, y_train)
results = [metrics(lin, X_train, X_test, y_train, y_test, "Линейная регрессия")]

# Доп. Задание
print("=== Дополнительные задания ===")

# 14) Boxplot MEDV
plt.figure(figsize=(7, 4))
sns.boxplot(x=df["MEDV"], orient="h")
plt.title("Boxplot MEDV")
plt.show()

q1, q3 = df["MEDV"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
mask_out = (df["MEDV"] < low) | (df["MEDV"] > high)
print(f"Q1={q1}, Q3={q3}, границы 'усов': [{low:.2f}, {high:.2f}]")
print("Выбросы по правилу 1.5*IQR:", sorted(df.loc[mask_out, "MEDV"].tolist()))
print("Из них значений MEDV = 50.0:", int((df["MEDV"] == 50.0).sum()), "\n")

# 15) Удаление выбросов, переобучение
df_clean = df[~mask_out]
print(f"Удалено строк: {int(mask_out.sum())}, осталось: {len(df_clean)}")

X_c, y_c = df_clean[features], df_clean["MEDV"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_c, y_c, test_size=0.25, random_state=RANDOM_STATE)
lin_c = LinearRegression().fit(Xc_train, yc_train)
results.append(metrics(lin_c, Xc_train, Xc_test, yc_train, yc_test,
                       "Линейная (без выбросов)"))

# 16) Ridge-регрессия
ridge = Ridge(alpha=1.0).fit(Xc_train, yc_train)
results.append(metrics(ridge, Xc_train, Xc_test, yc_train, yc_test,
                       "Ridge (alpha=1)"))

# 17) Полиномиальная регрессия, степень 3
poly = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly.fit(Xc_train, yc_train)
results.append(metrics(poly, Xc_train, Xc_test, yc_train, yc_test,
                       "Полином 3-й степени"))

print("\nИтоговая таблица:")
print(pd.DataFrame(results).round(3).to_string(index=False))
