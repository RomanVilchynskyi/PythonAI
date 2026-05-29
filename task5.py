import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)

speed = np.linspace(20, 140, 80)

fuel = 6 + 0.002*(speed - 65)**2

fuel += np.random.normal(0, 0.2, size=len(speed))

df = pd.DataFrame({
    "speed": speed,
    "fuel": fuel
})

df["time"] = 100 / df["speed"]

df["engine"] = np.random.randint(0, 2, size=len(df))

X = df[["speed", "time", "engine"]].values
y = df["fuel"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.fit(X_train, y_train, epochs=300, verbose=0)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)

new_data = np.array([
    [35, 100/35, 1],
    [95, 100/95, 1],
    [140, 100/140, 1]
])

new_data_scaled = scaler.transform(new_data)

predictions = model.predict(new_data_scaled)

print("\nПрогноз:")
for s, p in zip([35,95,140], predictions):
    print(f"{s} км/год -> {p[0]:.2f} л/100км")