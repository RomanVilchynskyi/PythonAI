import os
import pandas as pd
import numpy as np
import kagglehub

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Завантаження датасету
path = kagglehub.dataset_download("yasserh/titanic-dataset")
print("Path:", path)

# Зчитування CSV
csv_file = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(csv_file)

print(df.head())


# --------------------------
# Підготовка даних
# --------------------------
# Видаляємо непотрібні колонки
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# X - ознаки, y - ціль
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Числові та категоріальні ознаки
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

# Обробка числових
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Обробка категоріальних
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Об'єднання
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Підготовка
X = preprocessor.fit_transform(X)

# sparse -> dense
X = X.toarray() if hasattr(X, "toarray") else X

# train / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# numpy -> tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# --------------------------
# Нейронна мережа
# --------------------------
class TitanicNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


model = TitanicNN(X_train.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --------------------------
# Навчання
# --------------------------
epochs = 300

for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


# --------------------------
# Перевірка
# --------------------------
model.eval()

with torch.no_grad():
    predictions = model(X_test)
    predicted = (predictions >= 0.5).float()

    accuracy = (predicted == y_test).sum().item() / len(y_test)

print(f"\nAccuracy: {accuracy:.4f}")