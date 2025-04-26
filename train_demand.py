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

# Настройки из .env
from dotenv import load_dotenv
# 1) Загрузка .env
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
      time_date,
      COUNT(*) AS orders_count
    FROM orders
    GROUP BY restaurant_id, DATE(time_date), HOUR(time_date)
""", engine, parse_dates=['time_date'])

# 2. Фичи календаря
df['date']        = df['time_date'].dt.date
df['hour']        = df['time_date'].dt.hour
df['day_of_week'] = df['time_date'].dt.weekday
df['is_weekend']  = df['day_of_week'].isin([5,6]).astype(int)

# 3. Кодируем restaurant_id
le = LabelEncoder()
df['rest_enc'] = le.fit_transform(df['restaurant_id'])

# 4. Формируем X и y
X = df[['rest_enc','hour','day_of_week','is_weekend']]
y = df['orders_count']

# 5. train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Обучение XGBoost
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# 7. Оценка
preds = model.predict(X_val)
print("MAE:", mean_absolute_error(y_val, preds))

# 8. Сохраняем модель и энкодер
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/demand_model.pkl")
joblib.dump(le,    "models/restaurant_encoder.pkl")
print("Модель спроса сохранена в models/demand_model.pkl")
