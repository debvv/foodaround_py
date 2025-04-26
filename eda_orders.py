import pandas as pd
import matplotlib.pyplot as plt

# 1) Загрузка данных
df = pd.read_csv('orders.csv', parse_dates=['time_date'])
print("Список колонок в DataFrame:", df.columns.tolist())
# 2) Заказы по ресторанам
#orders_by_rest = df['restaurant_id'].value_counts().rename_axis('restaurant_id').reset_index(name='orders_count')
# старое — df['restaurant_id']
#fixed version down:
orders_by_rest = df['restaurant_id'].value_counts().rename_axis('restaurant_id').reset_index(name='orders_count')

print("Топ-10 ресторанов по числу заказов:")
print(orders_by_rest.head(10))

# 3) Распределение по датам
df['date'] = df['time_date'].dt.date
orders_by_date = df.groupby('date').size()
print("\nЗаказы по датам:")
print(orders_by_date.head())

# 4) Распределение по часам
df['hour'] = df['time_date'].dt.hour
orders_by_hour = df.groupby('hour').size()
print("\nЗаказы по часам:")
print(orders_by_hour)

# 5) Визуализация (одно окно)
plt.figure(figsize=(10, 4))
orders_by_hour.plot(kind='bar')
plt.title('Число заказов по часам')
plt.xlabel('Час дня')
plt.ylabel('Количество заказов')
plt.tight_layout()
plt.show()
