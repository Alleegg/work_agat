#время позднее но попросил ИИ протестировать модель, мог бы и сам но уже тяжело думать (12 ночи :( )

import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from linearRegression import LinearRegression

# Данные для сравнения
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 + 2 * X + np.random.randn(100, 1) * 2
y = y.ravel()  # делаем одномерным

my_model = LinearRegression()
my_model.fit(X, y)

# 2. Модель sklearn
sk_model = SklearnLinearRegression()
sk_model.fit(X, y)

# Сравнение коэффициентов
print("=== Сравнение коэффициентов ===")
print(f"Ваша модель: intercept = {my_model.coef[0]:.6f}, slope = {my_model.coef[1]:.6f}")
print(f"Sklearn:      intercept = {sk_model.intercept_:.6f}, slope = {sk_model.coef_[0]:.6f}")

# Проверка близости
coef_close = np.allclose(my_model.coef, [sk_model.intercept_, sk_model.coef_[0]])
print(f"Коэффициенты совпадают: {coef_close}")

# Сравнение предсказаний
y_pred_my = my_model.predict(X)
y_pred_sk = sk_model.predict(X)

print("\n=== Сравнение предсказаний ===")
print(f"Средняя разница: {np.mean(np.abs(y_pred_my - y_pred_sk)):.2e}")
print(f"Макс. разница:   {np.max(np.abs(y_pred_my - y_pred_sk)):.2e}")
print(f"Все предсказания близки: {np.allclose(y_pred_my, y_pred_sk)}")