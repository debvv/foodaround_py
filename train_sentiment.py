# Тренировка модели тональности

# Берёт scraped_reviews из БД, размечает «позитив/негатив».

#  Обучает LogisticRegression над TF-IDF, сохраняет models/sentiment_model.pkl.

import os
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL")

# 1. Загружаем отзывы
engine = create_engine(DB_URL)
reviews = pd.read_sql("SELECT review_text, rating FROM scraped_reviews", engine)
# Бинарная метка: 1=положительный(>=4), 0=отрицательный(<4)
reviews['sentiment'] = (reviews['rating'] >= 4).astype(int)

# 2. Обучаем Pipeline
model = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=1000)
)
model.fit(reviews['review_text'], reviews['sentiment'])
print("MAE not applicable, accuracy:", model.score(reviews['review_text'], reviews['sentiment']))

# 3. Сохраняем
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.pkl")
print("Sentiment модель сохранена в models/sentiment_model.pkl")
