# Тренировка модели спроса
# Загружает агрегированные исторические данные из БД или из datasets/orders.csv.
# Генерирует фичи («день недели», «час», one-hot restaurant_id)
# Обучает XGBoost/LightGBM, сохраняет модель в models/demand_model.pkl.

import os
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine
import holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
# Настройки из .env
from dotenv import load_dotenv
# === 1. Настройки из .env и подключение к БД ===
load_dotenv()

# 2) Получаем строку подключения из переменной окружения
#    Или пропишите её жёстко в виде строки:
# DB_URL = "mysql+pymysql://root:YOUR_PASSWORD@localhost/foodaround_db"
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("Переменная DB_URL не найдена в .env")

# 3) Создаём движок SQLAlchemy
engine = create_engine(DB_URL)

df = pd.read_sql("""
    SELECT 
      restaurant_id,
      DATE(time_date) AS date,
      COUNT(*) AS orders_count
    FROM orders
    GROUP BY restaurant_id, DATE(time_date)
""", engine, parse_dates=['date'])

# Конвертируем date в datetime для дальнейших признаков
df['time_date'] = pd.to_datetime(df['date'])
print("Всего строк после загрузки из БД:", len(df))

# 2.5. Добавляем пропущенные дни для каждого ресторана и заполняем 0
min_date = df['time_date'].min()
max_date = df['time_date'].max()
restaurants = df['restaurant_id'].unique()
full_idx = pd.MultiIndex.from_product(
    [restaurants, pd.date_range(min_date, max_date, freq='D')],
    names=['restaurant_id', 'time_date']
)
df = df.set_index(['restaurant_id', 'time_date']).reindex(full_idx).reset_index()
df['orders_count'] = df['orders_count'].fillna(0)

print("После расширения календаря (с нулями) строк:", len(df))

if df.empty:
    raise RuntimeError("Нет данных — проверьте таблицу orders в БД")



# === 3. Календарные признаки ===
df['day_of_week'] = df['time_date'].dt.weekday
df['is_weekend']  = df['day_of_week'].isin([5,6]).astype(int)


# === 4. Лаг-фичи (1–3 дня назад) ===
df = df.sort_values(['restaurant_id','time_date'])
for lag in (1,2,3):
    df[f'lag_{lag}'] = df.groupby('restaurant_id')['orders_count'].shift(lag)

# === 5. Сезонные признаки ===
df['month'] = df['time_date'].dt.month
rus_holidays = holidays.CountryHoliday('RU')
df['is_holiday'] = df['time_date'].dt.date.map(lambda d: 1 if d in rus_holidays else 0)


# === 6. Убираем NaN, получившиеся из-за лагов ===
df = df.dropna(subset=[f'lag_{l}' for l in (1,2,3)])
print("Строк после dropna:", len(df))
if len(df) < 2:
    raise RuntimeError("После удаления NaN слишком мало строк для обучения")



# === 7. Кодирование restaurant_id ===
le = LabelEncoder()
df['rest_enc'] = le.fit_transform(df['restaurant_id'])


# === 8. Формирование X и y ===
feature_cols = [
    'rest_enc', 'day_of_week', 'is_weekend',
    'lag_1', 'lag_2', 'lag_3',
    'month', 'is_holiday'
]
X = df[feature_cols]
y = df['orders_count']

# === 9. Хронологический train/test split ===
df = df.sort_values('time_date')
cut = int(len(df) * 0.8)
print(f"Train size: {cut}, Test size: {len(df)-cut}")

# === 9.1. Посмотрим, насколько много нулей и каких значений у нас в y ===
print("Распределение orders_count в train:")
print(y_train.value_counts(normalize=True).head(10))
print("\nРаспределение orders_count в test:")
print(y_test.value_counts(normalize=True).head(10))



X_train, y_train = X.iloc[:cut], y.iloc[:cut]
X_test,  y_test  = X.iloc[cut:], y.iloc[cut:]


# === 10. Обучение модели ===
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)
model.fit(X_train, y_train)

# === 11. Оценка модели на отложенной выборке ===
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Test MAE:  {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")


# === 11.1. Baseline: всегда берём значение lag_1 ===
baseline_preds = X_test['lag_1'].values
baseline_mae  = mean_absolute_error(y_test, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
print(f"Baseline MAE (lag_1):  {baseline_mae:.2f}")
print(f"Baseline RMSE (lag_1): {baseline_rmse:.2f}")




# === 12. Сохранение модели и энкодера ===
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/demand_model.pkl')
joblib.dump(le,    'models/restaurant_encoder.pkl')
print("Модель и энкодер сохранены в папке models/")

# === 13. Проверка загрузки модели (опционально) ===
loaded_model = joblib.load('models/demand_model.pkl')
loaded_encoder = joblib.load('models/restaurant_encoder.pkl')
print("Загруженная модель:", type(loaded_model))
print("Загруженный энкодер:", type(loaded_encoder))



