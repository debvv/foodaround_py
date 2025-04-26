import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1) загрузка .env и подключение к БД
load_dotenv()
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL не задана в .env")

engine = create_engine(DB_URL)

# 2) читаем скрапнутые отзывы с оценками
df = pd.read_sql("SELECT review_text, rating FROM scraped_reviews", engine)
df = df.dropna(subset=["review_text", "rating"])

# 3) готовим таргет: rating >= 3 → positive (1), иначе negative (0)
df["label"] = (df["rating"] >= 3).astype(int)

# 4) сплитим на train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["review_text"], df["label"], test_size=0.2, random_state=42
)

# 5) векторизация TF-IDF
vec = TfidfVectorizer(max_features=5000)
X_train_vec = vec.fit_transform(X_train)

# 6) обучаем простой классификатор
clf = LogisticRegression(max_iter=200, random_state=42)
clf.fit(X_train_vec, y_train)

# 7) оцениваем (опционально)
acc = clf.score(vec.transform(X_test), y_test)
print(f"Sentiment model accuracy: {acc:.3f}")

# 8) сохраняем в models/
os.makedirs("models", exist_ok=True)
joblib.dump(vec, "models/sentiment_vectorizer.pkl")
joblib.dump(clf, "models/sentiment_model.pkl")
print("Сохранено: models/sentiment_vectorizer.pkl, models/sentiment_model.pkl")
