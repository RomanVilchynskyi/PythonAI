import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

X = np.array([20,30,40,50,60,70,80,90,100,110]).reshape(-1,1)
y = np.array([9.0,7.8,6.9,6.5,6.3,6.4,6.7,7.3,8.0,9.2])

degrees = [2, 3, 4, 5]

best_mse = float("inf")
best_model = None
best_degree = None

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"Ступінь {d}: MSE={mse:.4f}, MAE={mae:.4f}")

    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_degree = d
        best_poly = poly

print("\nНайкращий ступінь:", best_degree)

# Прогноз
X_test = np.array([35, 95, 140]).reshape(-1,1)
X_test_poly = best_poly.transform(X_test)

predictions = best_model.predict(X_test_poly)

for i, val in enumerate(X_test.flatten()):
    print(f"Швидкість {val} км/год -> витрати {predictions[i]:.2f} л/100км")

# Графік
x_range = np.linspace(20, 150, 100).reshape(-1,1)
x_range_poly = best_poly.transform(x_range)
y_range = best_model.predict(x_range_poly)

plt.scatter(X, y, label="Дані")
plt.plot(x_range, y_range, label="Модель")
plt.legend()
plt.xlabel("Швидкість")
plt.ylabel("Витрати пального")
plt.title(f"Polynomial Regression (degree={best_degree})")
plt.show()