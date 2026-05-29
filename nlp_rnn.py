import pandas as pd
import torch
import torch.nn as nn

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from collections import Counter

# ==========================================
# 1. Завантаження датасету
# ==========================================

dataset = load_dataset("dair-ai/emotion")

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

# беремо тільки 3 емоції:
# 0 = sadness
# 3 = anger
# 1 = joy

allowed_labels = [0, 1, 3]

filtered_texts = []
filtered_labels = []

for text, label in zip(train_texts, train_labels):

    if label in allowed_labels:

        filtered_texts.append(text)
        filtered_labels.append(label)

# ==========================================
# 2. Перетворення labels
# ==========================================

label_map = {
    0: "sadness",
    1: "joy",
    3: "anger"
}

string_labels = []

for label in filtered_labels:
    string_labels.append(label_map[label])

encoder = LabelEncoder()

encoded_labels = encoder.fit_transform(string_labels)

# ==========================================
# 3. Токенізація
# ==========================================

all_words = []

for text in filtered_texts:

    words = text.lower().split()

    for word in words:
        all_words.append(word)

counter = Counter(all_words)

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

index = 2

for word, count in counter.items():

    if count >= 2:
        vocab[word] = index
        index += 1

# ==========================================
# 4. Текст -> числа
# ==========================================

max_len = 15

sequences = []

for text in filtered_texts:

    words = text.lower().split()

    seq = []

    for word in words:

        if word in vocab:
            seq.append(vocab[word])
        else:
            seq.append(vocab["<UNK>"])

    # padding
    if len(seq) < max_len:

        while len(seq) < max_len:
            seq.append(vocab["<PAD>"])

    else:
        seq = seq[:max_len]

    sequences.append(seq)

# ==========================================
# 5. Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    sequences,
    encoded_labels,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 6. Dataset
# ==========================================

class EmotionDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        return self.X[index], self.y[index]

train_dataset = EmotionDataset(X_train, y_train)
test_dataset = EmotionDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)

# ==========================================
# 7. RNN модель
# ==========================================

class RNNModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        embedded = self.embedding(x)

        output, hidden = self.rnn(embedded)

        hidden = hidden.squeeze(0)

        out = self.fc(hidden)

        return out

model = RNNModel(
    vocab_size=len(vocab),
    embed_size=64,
    hidden_size=128,
    num_classes=3
)

# ==========================================
# 8. Loss + Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================================
# 9. Навчання
# ==========================================

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# ==========================================
# 10. Тестування
# ==========================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == y_batch).sum().item()

        total += y_batch.size(0)

accuracy = correct / total

print("\nAccuracy:", accuracy)

# ==========================================
# 11. Перевірка на нових фразах
# ==========================================

def predict_emotion(text):

    words = text.lower().split()

    seq = []

    for word in words:

        if word in vocab:
            seq.append(vocab[word])
        else:
            seq.append(vocab["<UNK>"])

    if len(seq) < max_len:

        while len(seq) < max_len:
            seq.append(vocab["<PAD>"])

    else:
        seq = seq[:max_len]

    tensor = torch.tensor([seq], dtype=torch.long)

    with torch.no_grad():

        output = model(tensor)

        prediction = torch.argmax(output, dim=1).item()

    emotion = encoder.inverse_transform([prediction])[0]

    return emotion

# ==========================================
# 12. Приклади
# ==========================================

examples = [
    "I am very happy today",
    "I hate everything",
    "I feel so lonely and sad"
]

for text in examples:

    emotion = predict_emotion(text)

    print(f"\nText: {text}")
    print(f"Emotion: {emotion}")