import numpy as np
import matplotlib.pyplot as plt


# Завдання 1
x = np.linspace(-10, 10, 1000)
y = x**2 * np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("f(x) = x^2 * sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()


# Завдання 2

data = np.random.normal(loc=5, scale=2, size=1000)

plt.figure()
plt.hist(data, bins=30)
plt.title("Гістограма нормального розподілу")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.show()


# Завдання 3

hobbies = ["Gaming", "Coding", "Music", "Sport", "Movies"]
values = [30, 25, 15, 20, 10]

plt.figure()
plt.pie(values, labels=hobbies, autopct='%1.1f%%')
plt.title("Мої хобі")
plt.show()


# Завдання 4

apple = np.random.normal(150, 10, 100)
banana = np.random.normal(120, 15, 100)
orange = np.random.normal(130, 12, 100)
pear = np.random.normal(140, 8, 100)

plt.figure()
plt.boxplot([apple, banana, orange, pear], labels=["Apple", "Banana", "Orange", "Pear"])
plt.title("Box-plot маси фруктів")
plt.ylabel("Грами")
plt.show()