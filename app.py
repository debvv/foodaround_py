import os
import json
import subprocess
from datetime import datetime

from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv

import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

import joblib
import numpy as np
from surprise import SVD

from sklearn.metrics.pairwise import cosine_similarity

import gensim
from gensim.corpora import Dictionary

# ============ CONFIG ==============
load_dotenv()
DB_URL   = os.getenv("DB_URL")   # "mysql+pymysql://root:pass@localhost/foodaround_db"
DB_HOST  = os.getenv("DB_HOST",  "localhost")
DB_USER  = os.getenv("DB_USER",  "root")
DB_PASS  = os.getenv("DB_PASS",  "")
DB_NAME  = os.getenv("DB_NAME",  "foodaround_db")

MODEL_DIR = "models"

# SQLAlchemy engine для pd.read_sql
engine = create_engine(DB_URL, pool_pre_ping=True)

# Готовые модели (лениво, но один раз загрузятся)
demand_model       = joblib.load(os.path.join(MODEL_DIR, "demand_model.pkl"))
restaurant_encoder = joblib.load(os.path.join(MODEL_DIR, "restaurant_encoder.pkl"))

cf_model           = joblib.load(os.path.join(MODEL_DIR, "cf_model.pkl"))          # Surprise SVD
tfidf_vec          = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
tfidf_matrix       = joblib.load(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"))

sentiment_vec      = joblib.load(os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl"))
sentiment_model    = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))

lda_model          = gensim.models.LdaModel.load(os.path.join(MODEL_DIR, "lda_model.gensim"))
lda_dictionary     = Dictionary.load(os.path.join(MODEL_DIR, "lda_dictionary.gensim"))

# ============ APP ==============
app = Flask(__name__)

def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset='utf8mb4',
        use_unicode=True,
        use_pure=True
    )

@app.route("/ping")
def ping():
    return "pong"

@app.route('/get_restaurants', methods=['GET'])
def get_restaurants():
    df = pd.read_sql(
        "SELECT id, name, address, rating, cuisine FROM restaurants",
        con=engine
    )
    return jsonify(restaurants=df.to_dict(orient='records'))

@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    """
    Input JSON:
      { "restaurant_id": int, "date_time": ISO8601 str }
    Output:
      { "prediction": float }
    """
    j = request.json
    rid = j.get("restaurant_id")
    dt  = j.get("date_time")
    if rid is None or dt is None:
        return jsonify(error="restaurant_id and date_time required"), 400

    # parse date_time
    ts = datetime.fromisoformat(dt)
    # build features
    enc = restaurant_encoder.transform([rid])[0]
    hour = ts.hour
    dow  = ts.weekday()
    is_we = 1 if dow >= 5 else 0

    X = np.array([[enc, hour, dow, is_we]])
    pred = float(demand_model.predict(X))
    return jsonify(prediction=pred)

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Input JSON:
      { "user_id": int, "n": int }
    Output:
      {
        "cf": [restaurant_id...],
        "cb": [restaurant_id...]
      }
    """
    j = request.json
    uid = j.get("user_id")
    n   = int(j.get("n", 5))
    if uid is None:
        return jsonify(error="user_id required"), 400

    # --- Collaborative: предскажем по SVD ---
    # узнаём все рестораны
    df_r = pd.read_sql("SELECT id FROM restaurants", con=engine)
    all_ids = df_r["id"].tolist()

    # CF: предскажем для тех, что юзер ещё не брал
    # узнаём что уже заказал
    df_o = pd.read_sql(
        f"SELECT restaurant_id FROM orders WHERE from_id = {uid}",
        con=engine
    )
    seen = set(df_o["restaurant_id"].tolist())
    candidates = [rid for rid in all_ids if rid not in seen]

    preds = []
    for rid in candidates:
        est = cf_model.predict(str(uid), str(rid)).est
        preds.append((rid, est))
    cf_top = [rid for rid,_ in sorted(preds, key=lambda x:-x[1])[:n]]

    # --- Content-based ---
    # найдём для юзера самое популярное заведение
    df_user = pd.read_sql(
        f"SELECT restaurant_id, SUM(count) as cnt FROM orders WHERE from_id = {uid} GROUP BY restaurant_id",
        con=engine
    )
    if df_user.empty:
        cb_top = []
    else:
        fave = int(df_user.sort_values("cnt", ascending=False).iloc[0]["restaurant_id"])
        # найдём index в матрице TF-IDF
        df_r = pd.read_sql("SELECT id FROM restaurants", con=engine)
        idx = df_r.index[df_r["id"] == fave].tolist()[0]
        cos = cosine_similarity(
            tfidf_matrix[idx:idx+1],
            tfidf_matrix
        ).flatten()
        # получаем топ N похожих (кроме самого)
        pairs = list(enumerate(cos))
        pairs.sort(key=lambda x:-x[1])
        cb_top = [int(df_r.iloc[i]["id"]) for i,sim in pairs[1:n+1]]

    return jsonify(cf=cf_top, cb=cb_top)

@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    """
    Input JSON:
      { "review_text": str }
    Output:
      { "sentiment": "positive"/"negative" }
    """
    text = request.json.get("review_text", "")
    if not text:
        return jsonify(error="review_text required"), 400
    v = sentiment_vec.transform([text])
    pred = sentiment_model.predict(v)[0]
    label = "positive" if pred==1 else "negative"
    return jsonify(sentiment=label)

@app.route('/topic_reviews', methods=['POST'])
def topic_reviews():
    """
    Input JSON:
      { "review_text": str }
    Output:
      { "topics": [ {"topic_id":int, "score":float}, ... ] }
    """
    text = request.json.get("review_text", "")
    if not text:
        return jsonify(error="review_text required"), 400
    toks = text.lower().split()
    bow = lda_dictionary.doc2bow(toks)
    topics = lda_model.get_document_topics(bow)
    # берём топ-3
    top3 = sorted(topics, key=lambda x:-x[1])[:3]
    return jsonify(topics=[{"topic_id": int(t), "score": float(sc)} for t,sc in top3])

@app.route('/update_data', methods=['POST'])
def update_data():
    """
    Триггерит пересборку моделей:
      train_demand.py, train_recommender.py, train_sentiment.py, train_topic_model.py
    """
    scripts = [
        "train_demand.py",
        "train_recommender.py",
        "train_sentiment.py",
        "train_topic_model.py"
    ]
    results = {}
    for s in scripts:
        try:
            out = subprocess.check_output(
                ["python", s],
                cwd=os.getcwd(),
                stderr=subprocess.STDOUT,
                text=True
            )
            results[s] = {"ok": True,  "output": out}
        except subprocess.CalledProcessError as e:
            results[s] = {"ok": False, "output": e.output}
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
