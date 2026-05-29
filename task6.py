import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

print("Назви сортів вина:", target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(16, 12),
    activation='relu',
    max_iter=500,
    random_state=42,
    verbose=True
)

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=target_names))

print("\nПриклади передбачень:")
sample_indices = [0, 10, 2]

for idx in sample_indices:
    pred = mlp.predict(X_test[idx].reshape(1, -1))[0]
    actual = y_test[idx]
    print(f"Sample №{idx}: Predicted: {target_names[pred]}, Real: {target_names[actual]}")