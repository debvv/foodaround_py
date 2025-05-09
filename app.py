import os
import json
from datetime import datetime
from sqlalchemy import text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mysql.connector
from sqlalchemy import create_engine
from dotenv import load_dotenv
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.utils import simple_preprocess
from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import holidays
load_dotenv()

# Настройки
DB_URL       = os.getenv("DB_URL")  # например "mysql+pymysql://root:pass@localhost/foodaround_db"
# ЗАГРУЗКА LDA Данных
MODEL_DIR    = "models"
lda_dictionary = corpora.Dictionary.load(os.path.join(MODEL_DIR, "lda_dictionary.gensim"))
lda_model      = LdaModel.load(os.path.join(MODEL_DIR, "lda_model.gensim"))

# 2) определяем preprocess (должен совпадать с тем, что вы юзали при train_topic_model.py) ---

nltk.download("stopwords")
_ru = stopwords.words("russian")
_en = stopwords.words("english")
#STOPWORDS = set(stopwords.words("russian"))
#LEMMA = WordNetLemmatizer()
STOPWORDS = set(_ru + _en)

# Flask
app = Flask(__name__)

# === 2. Загрузка модели и энкодера ===
model   = joblib.load('models/demand_model.pkl')
encoder = joblib.load('models/restaurant_encoder.pkl')

# Создаём объект праздников
rus_hols = holidays.CountryHoliday('RU')

CORS(app)



# SQLAlchemy-движок (для чтения через pandas)
engine = create_engine(DB_URL)

# mysql-connector (для вставок и CRUD)
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # ваш пароль
        database="foodaround_db",
        charset='utf8mb4',
        use_unicode=True,
        use_pure=True
    )
# ---------------------------

#добавил в app.py ту же функцию предобработки текстов, которую использовал при тренировке LDA,
# и подключил её перед вызовом analyze_topics.


def preprocess(text: str) -> list[str]:
    # простая токенизация + удаление стоп-слов
    tokens = simple_preprocess(text, deacc=True)  # деакцентирование и lower
    return [t for t in tokens if t not in STOPWORDS]


# --------------- Загрузка моделей ---------------

# Спрос
demand_model        = joblib.load(os.path.join(MODEL_DIR, "demand_model.pkl"))
restaurant_encoder  = joblib.load(os.path.join(MODEL_DIR, "restaurant_encoder.pkl"))

# Рекомендации CF + CB
cf_model            = joblib.load(os.path.join(MODEL_DIR, "cf_model.pkl"))
tfidf               = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
tfidf_matrix        = joblib.load(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"))

# Отзывы: сентимент
sentiment_model     = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))
sentiment_vectorizer= joblib.load(os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl"))

# Темы
lda_dict            = Dictionary.load(os.path.join(MODEL_DIR, "lda_dictionary.gensim"))
lda_model           = LdaModel.load(os.path.join(MODEL_DIR, "lda_model.gensim"))

model = joblib.load('models/xgb_model.pkl')

# Подгружаем список ресторанов сразу (для /recommend и /get_restaurants)
restaurants_df = pd.read_sql("SELECT id, name, address, rating, cuisine FROM restaurants", engine)


# --------------- Эндпоинты ---------------

@app.route("/ping")
def ping():
    return "pong"


@app.route('/get_restaurants', methods=['GET'])
def get_restaurants():
    # уже есть restaurants_df
    resp = restaurants_df[["id","name","address","rating","cuisine"]].to_dict(orient="records")
    return jsonify({"restaurants": resp})


@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    j = request.json
    rid = j.get("restaurant_id")
    hour = j.get("hour")
    dow  = j.get("day_of_week")
    weekend = j.get("is_weekend")
    # проверим, что ресторан в энкодере есть
    if rid not in restaurant_encoder.classes_:
        return jsonify({"error": "Unknown restaurant_id"}), 400

    enc = int(restaurant_encoder.transform([rid])[0])
    X = np.array([[enc, hour, dow, weekend]])
    pred = float(demand_model.predict(X)[0])
    return jsonify({"prediction": pred})


# функция передачи фичей и ендпоинт -------------------------------------------------------

# === 3. Функция генерации фич для одного ресторана и одной даты ===

def generate_features(date_str: str, restaurant_id: int) -> np.ndarray:
    # 1) Преобразуем входную строку в datetime
    dt = pd.to_datetime(date_str)
    date = dt.date()

    # 2) История из БД
    query = f"""
        SELECT
            DATE(time_date) AS date,
            COUNT(*) AS orders_count
        FROM orders
        WHERE restaurant_id = {restaurant_id}
        GROUP BY DATE(time_date)
        ORDER BY date
    """
    df_hist = pd.read_sql(query, engine, parse_dates=['date'])
    df_hist = df_hist.set_index('date').asfreq('D').fillna(0)

    # 3) Лаги
    lag1 = df_hist['orders_count'].shift(1).get(date, 0)
    lag2 = df_hist['orders_count'].shift(2).get(date, 0)
    lag3 = df_hist['orders_count'].shift(3).get(date, 0)

    # 4) Календарные фичи
    hour = dt.hour             # Единичный признак часа (для суточной модели всегда 0)
    dow   = dt.weekday()       # день недели
    is_we = int(dow >= 5)      # выходной
    month = dt.month           # месяц
    is_hol= int(date in rus_hols)  # праздник?

    # 5) Код ресторана
    rest_enc = encoder.transform([restaurant_id])[0]

    # 6) Собираем всё в тот же порядок, что и при обучении
    features = [
        rest_enc,
        hour,
        dow,
        is_we,
        lag1,
        lag2,
        lag3,
        month,
        is_hol
    ]
    return np.array(features).reshape(1, -1)



# === 4. Эндпоинт /predict ===
@app.route('/predict', methods=['GET'])
def predict():
    date_str      = request.args.get('date')
    restaurant_id = request.args.get('restaurant_id')
    if not date_str or not restaurant_id:
        return jsonify({"error": "Нужно указать date и restaurant_id"}), 400

    restaurant_id = int(restaurant_id)
    feats = generate_features(date_str, restaurant_id)
    pred  = model.predict(feats)[0]

    return jsonify({
        "date": date_str,
        "restaurant_id": restaurant_id,
        "predicted_orders": float(pred)
    })


# --------------------------------------------------------------------------------------


@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json.get("user_id")
    if user_id is None:
        return jsonify({"error": "Missing user_id"}), 400

    # --- Collaborative recommendations (SVD) ---
    all_rest_ids = restaurants_df["id"].astype(str).tolist()
    cf_scores = [
        (int(rid), cf_model.predict(str(user_id), rid).est)
        for rid in all_rest_ids
    ]
    cf_scores.sort(key=lambda x: x[1], reverse=True)
    top_cf_ids   = [rid for rid,_ in cf_scores[:5]]
    cf_names     = restaurants_df.set_index("id").loc[top_cf_ids, "name"].tolist()

    # --- Content-based recommendations (cosine on TF-IDF) ---
    # получаем последний заказ
    q = text("""
        SELECT restaurant_id
        FROM orders
        WHERE from_id = :uid
        ORDER BY time_date DESC
        LIMIT 1
    """)
    conn = engine.connect()
    last = conn.execute(q, {"uid": user_id}).fetchone()
    conn.close()

    if last:
        last_rest = last[0]
        # находим индекс в матрице TF-IDF
        idx = restaurants_df.index[restaurants_df["id"] == last_rest][0]
        sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        # топ-5 похожих (пропускаем сам себя)
        top_idxs = sims.argsort()[::-1][1:6]
        cb_names = restaurants_df.iloc[top_idxs]["name"].tolist()
    else:
        # если пользователь без истории — просто первые 5 ресторанов
        cb_names = restaurants_df["name"].tolist()[:5]

    return jsonify({
        "cf_recommendations": cf_names,
        "cb_recommendations": cb_names
    })

@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    text = request.json.get("review_text","")
    vec = sentiment_vectorizer.transform([text])
    pred = sentiment_model.predict(vec)[0]
    sentiment = "positive" if pred==1 else "negative"
    return jsonify({"sentiment": sentiment})


#3 version
@app.route('/analyze_topics', methods=['POST'])
def analyze_topics():
    text = request.json.get("review_text", "").strip()
    if not text:
        return jsonify({"error": "Missing review_text"}), 400

    # 1) токенизируем и фильтруем стоп-слова
    tokens = preprocess(text)

    # 2) строим «мешок слов» для LDA
    bow = lda_dictionary.doc2bow(tokens)

    # 3) получаем распределение по топикам
    topics = lda_model.get_document_topics(bow)

    # 4) возвращаем результат
    return jsonify({"topics": [
        {"topic_id": tid, "probability": float(prob)}
        for tid, prob in topics
    ]})




# -- Существующие CRUD для отзывов --

@app.route('/add_chef_review', methods=['POST'])
def add_chef_review():
    data = request.json
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SET NAMES utf8mb4 COLLATE utf8mb4_general_ci")
    cur.execute(
        "INSERT INTO chef_reviews (chef_name,rating,comment,created_at) VALUES (%s,%s,%s,NOW())",
        (data["chef_name"], data["rating"], data["comment"])
    )
    conn.commit()
    rid = cur.lastrowid
    cur.close(); conn.close()
    return jsonify({"message":"Отзыв добавлен","review_id":rid})


@app.route('/get_chef_reviews', methods=['GET'])
def get_chef_reviews():
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SET NAMES utf8mb4 COLLATE utf8mb4_general_ci")
    cur.execute("SELECT id, chef_name,rating,comment FROM chef_reviews")
    rows = cur.fetchall()
    cur.close(); conn.close()
    data = [{"id":r[0],"chef_name":r[1],"rating":r[2],"comment":r[3]} for r in rows]
    return jsonify({"chef_reviews": data})


@app.route('/add_scraped_review', methods=['POST'])
def add_scraped_review():
    d = request.json
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO scraped_reviews (restaurant_name,source,rating,review_text,review_date,created_at) "
        "VALUES (%s,%s,%s,%s,%s,NOW())",
        (d["restaurant_name"],d["source"],d["rating"],d["review_text"],d["review_date"])
    )
    conn.commit()
    rid = cur.lastrowid
    cur.close(); conn.close()
    return jsonify({"message":"Скрап-отзыв добавлен","review_id":rid})


@app.route('/get_scraped_reviews', methods=['GET'])
def get_scraped_reviews():
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT id,restaurant_name,source,rating,review_text,review_date FROM scraped_reviews")
    rows = cur.fetchall()
    cur.close(); conn.close()
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


@app.route('/update_data', methods=['POST'])
def update_data():
    # здесь по необходимости можно запустить обновление модели, повторный train_*.py и т.п.
    return jsonify({"status":"Данные обновлены"})


# --------------- Запуск ---------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
