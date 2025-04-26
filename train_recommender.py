# Тренировка рекомендателя
#  Content-based: собирает TF-IDF по названиям + кухням, считает косинус-сходство.
# Collaborative: использует surprise.SVD на выборках (user, restaurant, count).
# Сохраняет оба варианта в models/cb_model.pkl и models/cf_model.pkl.

import os
import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL")

# Подключаемся и загружаем таблицы
engine = create_engine(DB_URL)
orders = pd.read_sql("SELECT from_id, restaurant_id, count FROM orders", engine)
restaurants = pd.read_sql("SELECT id, name, cuisine FROM restaurants", engine)

# ----- Collaborative Filtering (SVD) -----
reader = Reader(rating_scale=(orders['count'].min(), orders['count'].max()))
data = Dataset.load_from_df(orders[['from_id','restaurant_id','count']], reader)
trainset = data.build_full_trainset()
algo = SVD(n_factors=50, random_state=42)
algo.fit(trainset)


# Сохраняем CF-модель
os.makedirs("models", exist_ok=True)
joblib.dump(algo, "models/cf_model.pkl")
print("CF-модель сохранена в models/cf_model.pkl")


# ----- Content-Based Filtering (TF-IDF) -----
tfidf = TfidfVectorizer(max_features=5000)
# Склеиваем name + cuisine для текста
docs = (restaurants['name'].fillna('') + ' ' + restaurants['cuisine'].fillna(''))
matrix = tfidf.fit_transform(docs)

# Сохраняем TF-IDF и матрицу
joblib.dump(tfidf,   "models/tfidf_vectorizer.pkl")
joblib.dump(matrix,  "models/tfidf_matrix.pkl")
print("TF-IDF модели сохранены в models/")
