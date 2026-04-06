import pandas as pd

# 1. Створення даних
data = {
    "OrderID": [1001, 1002, 1003],
    "Customer": ["Alice", "Bob", "Alice"],
    "Product": ["Laptop", "Chair", "Mouse"],
    "Category": ["Electronics", "Furniture", "Electronics"],
    "Quantity": [1, 2, 3],
    "Price": [1500, 180, 25],
    "OrderDate": ["2023-06-01", "2023-06-03", "2023-06-05"]
}

df = pd.DataFrame(data)

# Конвертація дати
df["OrderDate"] = pd.to_datetime(df["OrderDate"])

# 2. Новий стовпець
df["TotalAmount"] = df["Quantity"] * df["Price"]

# 3a. Сумарний дохід
total_income = df["TotalAmount"].sum()
print("Сумарний дохід:", total_income)

# 3b. Середнє значення
avg_total = df["TotalAmount"].mean()
print("Середній чек:", avg_total)

# 3c. Кількість замовлень по клієнтах
orders_per_customer = df["Customer"].value_counts()
print("\nКількість замовлень:\n", orders_per_customer)

# 4. Замовлення > 500
print("\nЗамовлення > 500:")
print(df[df["TotalAmount"] > 500])

# 5. Сортування по даті (спадання)
print("\nСортування по даті:")
print(df.sort_values(by="OrderDate", ascending=False))

# 6. Замовлення з 5 по 10 червня
filtered_dates = df[(df["OrderDate"] >= "2023-06-05") & (df["OrderDate"] <= "2023-06-10")]
print("\nЗамовлення 5-10 червня:")
print(filtered_dates)

# 7. Групування по категорії
grouped = df.groupby("Category").agg({
    "Quantity": "sum",
    "TotalAmount": "sum"
})
print("\nГрупування по категорії:")
print(grouped)

# 8. ТОП-3 клієнти
top_customers = df.groupby("Customer")["TotalAmount"].sum().sort_values(ascending=False).head(3)
print("\nТОП-3 клієнти:")
print(top_customers)

df.to_csv("orders.csv", index=False)