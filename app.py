import os
import json
import joblib
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, Response
from sqlalchemy import create_engine
import mysql.connector

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

# ----------------------------------------------------------------------
# 1) Конфигурирование и загрузка моделей
# ----------------------------------------------------------------------
load_dotenv()  # загружаем переменные из .env
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL не задана в .env")

MODEL_DIR = "models"

# SQLAlchemy-движок для pd.read_sql
engine = create_engine(DB_URL)

# Flask-приложение
app = Flask(__name__)

# Функция для прямых INSERT/SELECT, если нужно
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", "foodaround_db"),
        charset="utf8mb4",
        use_unicode=True,
        use_pure=True
    )

# --- Demand Forecasting ---
demand_model      = joblib.load(os.path.join(MODEL_DIR, "demand_model.pkl"))
restaurant_encoder = joblib.load(os.path.join(MODEL_DIR, "restaurant_encoder.pkl"))

# --- Collaborative Filtering ---
cf_model = joblib.load(os.path.join(MODEL_DIR, "cf_model.pkl"))
# заранее читаем список ресторанов из БД
restaurants_df = pd.read_sql("SELECT id, name FROM restaurants", engine)
restaurant_ids = restaurants_df["id"].tolist()

# --- Content-Based Filtering ---
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
tfidf_matrix     = joblib.load(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"))
# Подтягиваем метаданные для CB
restaurants_meta = pd.read_sql("SELECT id, name, cuisine FROM restaurants", engine)

# --- Sentiment Analysis ---
sentiment_vectorizer = joblib.load(os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl"))
sentiment_model      = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))

# --- Topic Modeling ---
lda_dict  = Dictionary.load(os.path.join(MODEL_DIR, "lda_dictionary.gensim"))
lda_model = LdaModel.load(os.path.join(MODEL_DIR, "lda_model.gensim"))

# ----------------------------------------------------------------------
# 2) Эндпоинты
# ----------------------------------------------------------------------

@app.route("/ping")
def ping():
    return "pong"

# --- Restaurants list ---
@app.route('/get_restaurants', methods=['GET'])
def get_restaurants():
    rows = restaurants_df.to_dict(orient="records")
    return jsonify({"restaurants": rows})

# --- Demand Prediction ---
@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    data = request.json or {}
    rid  = data.get("restaurant_id")
    hour = data.get("hour")
    dow  = data.get("day_of_week")
    is_we = data.get("is_weekend")
    if rid is None or hour is None or dow is None or is_we is None:
        return jsonify({"error": "Неполные входные данные"}), 400

    # кодируем ресторан
    try:
        enc = restaurant_encoder.transform([rid])[0]
    except ValueError:
        return jsonify({"error": "Unknown restaurant_id"}), 400

    X = np.array([[enc, hour, dow, is_we]])
    pred = demand_model.predict(X)[0]
    return jsonify({"prediction": float(pred)})

# --- Recommendations ---
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json.get("user_id")
    if user_id is None:
        return jsonify({"error": "user_id не указан"}), 400

    # Collaborative: предсказываем рейтинг для каждого ресторана
    preds = []
    for rid in restaurant_ids:
        est = cf_model.predict(user_id, rid).est
        preds.append((rid, est))
    preds.sort(key=lambda x: -x[1])
    top_cf = preds[:5]
    cf_names = [restaurants_df.loc[restaurants_df.id==rid, "name"].values[0] for rid,_ in top_cf]

    # Content-based: cosine similarity к каждому ресторану
    # например, средний профиль пользователя можем получить как average TF-IDF
    # в данном примере просто берем top_cf[0]
    tfidf_user = tfidf_matrix[restaurant_ids.index(top_cf[0][0])]
    sims = cosine_similarity(tfidf_user, tfidf_matrix).flatten()
    top_cb_idx = sims.argsort()[::-1][1:6]
    cb_names = [restaurants_meta.loc[idx, "name"] for idx in top_cb_idx]

    return jsonify({
        "cf_recommendations": cf_names,
        "cb_recommendations": cb_names
    })

# --- Sentiment Analysis ---
@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    text = request.json.get("review_text", "")
    vec  = sentiment_vectorizer.transform([text])
    label = sentiment_model.predict(vec)[0]
    sentiment = "positive" if label == 1 else "negative"
    return jsonify({"sentiment": sentiment})

# --- Topic Modeling ---
@app.route('/analyze_topics', methods=['POST'])
def analyze_topics():
    text = request.json.get("review_text", "").lower().split()
    bow  = lda_dict.doc2bow(text)
    topics = lda_model.get_document_topics(bow, minimum_probability=0.05)
    top_n = sorted(topics, key=lambda x: -x[1])[:3]
    out = [{"topic_id": int(tid), "prob": float(prob)} for tid,prob in top_n]
    return jsonify({"topics": out})

# --- Chef Reviews CRUD ---
@app.route('/add_chef_review', methods=['POST'])
def add_chef_review():
    data = request.json or {}
    name    = data.get("chef_name")
    rating  = data.get("rating")
    comment = data.get("comment")
    if not all([name, rating, comment]):
        return jsonify({"error": "Неполные данные"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chef_reviews (chef_name, rating, comment, created_at) "
                   "VALUES (%s, %s, %s, NOW())",
                   (name, rating, comment))
    conn.commit()
    rid = cursor.lastrowid
    cursor.close(); conn.close()
    return jsonify({"message": "Отзыв добавлен", "review_id": rid})

@app.route('/get_chef_reviews', methods=['GET'])
def get_chef_reviews():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, chef_name, rating, comment FROM chef_reviews")
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    out = [{"id": r[0], "chef_name": r[1], "rating": r[2], "comment": r[3]} for r in rows]
    return jsonify({"chef_reviews": out})

# --- Scraped Reviews CRUD ---
@app.route('/add_scraped_review', methods=['POST'])
def add_scraped_review():
    d = request.json or {}
    fields = ("restaurant_name","source","rating","review_text","review_date")
    if not all(d.get(f) for f in fields):
        return jsonify({"error": "Неполные данные"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scraped_reviews (restaurant_name, source, rating, review_text, review_date, created_at) "
        "VALUES (%s,%s,%s,%s,%s,NOW())",
        tuple(d[f] for f in fields)
    )
    conn.commit()
    rid = cursor.lastrowid
    cursor.close(); conn.close()
    return jsonify({"message": "Скрапнутый отзыв добавлен", "review_id": rid})

@app.route('/get_scraped_reviews', methods=['GET'])
def get_scraped_reviews():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, restaurant_name, source, rating, review_text, review_date FROM scraped_reviews")
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "restaurant_name": r[1],
            "source": r[2],
            "rating": r[3],
            "review_text": r[4],
            "review_date": str(r[5])
        })
    return jsonify({"scraped_reviews": out})

# ----------------------------------------------------------------------
# 3) Запуск
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
