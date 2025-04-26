from flask import Flask, request, jsonify, Response
import json
import mysql.connector
import joblib
import gensim
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ------------------
# 1) Функция для подключения к MySQL
# ------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="foodaround_db",
        charset='utf8mb4',
        use_unicode=True,
        use_pure=True
    )

# ------------------
# 2) Загрузка моделей и вспомогательных объектов
# ------------------
# Прогноз спроса
demand_model = joblib.load("models/demand_model.pkl")
rest_encoder = joblib.load("models/restaurant_encoder.pkl")

# Рекомендации
cf_model     = joblib.load("models/cf_model.pkl")
tfidf_vect   = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

# Анализ тональности
sentiment_model = joblib.load("models/sentiment_model.pkl")

# Тематическое моделирование
lda_model = gensim.models.LdaModel.load("models/lda_model.gensim")
lda_dict  = gensim.corpora.Dictionary.load("models/lda_dictionary.gensim")

# Подгружаем DataFrame ресторанов для CF и CB
conn = get_db_connection()
restaurants_df = pd.read_sql("SELECT id, name, cuisine FROM restaurants", conn)
conn.close()


# ------------------
# 3) Эндпоинты
# ------------------

# 3.1 Прогноз спроса
@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    data = request.json
    rest_id     = int(data['restaurant_id'])
    order_date  = data['order_date']   # e.g. "2025-05-02"
    order_hour  = int(data['order_hour'])  # e.g. 17

    # Фичи календаря
    dt = pd.to_datetime(order_date)
    day_of_week = dt.weekday()
    is_weekend  = int(day_of_week >= 5)

    # Кодируем ресторан
    rest_enc = rest_encoder.transform([rest_id])[0]

    # Собираем вектор признаков
    X = np.array([[rest_enc, order_hour, day_of_week, is_weekend]])

    # Предсказываем
    pred = demand_model.predict(X)[0]
    return jsonify({'prediction': float(pred)})


# 3.2 Система рекомендаций
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.json['user_id'])

    # Collaborative filtering
    all_ids = restaurants_df['id'].tolist()
    cf_preds = [(rid, cf_model.predict(user_id, rid).est) for rid in all_ids]
    cf_top   = sorted(cf_preds, key=lambda x: x[1], reverse=True)[:5]

    # Content-based
    conn = get_db_connection()
    user_orders = pd.read_sql(f"SELECT restaurant_id FROM orders WHERE from_id={user_id}", conn)
    conn.close()
    cb_top = []
    if not user_orders.empty:
        vecs = tfidf_matrix[user_orders['restaurant_id'].values]
        user_vec = vecs.mean(axis=0)
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        idx  = np.argsort(sims)[-5:][::-1]
        cb_top = [(int(restaurants_df.iloc[i]['id']), float(sims[i])) for i in idx]

    return jsonify({'cf_recommendations': cf_top, 'cb_recommendations': cb_top})


# 3.3 Сентимент-анализ отзывов
@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    text = request.json.get('review_text', '')
    pred = sentiment_model.predict([text])[0]
    proba = sentiment_model.predict_proba([text])[0].tolist()
    return jsonify({'sentiment': int(pred), 'probability': proba})


# 3.4 Тематическое моделирование
@app.route('/topic_reviews', methods=['POST'])
def topic_reviews():
    text = request.json.get('review_text', '').lower().split()
    bow  = lda_dict.doc2bow(text)
    topics = lda_model.get_document_topics(bow)
    return jsonify({'topics': topics})


# ------------------
# 4) Ваши существующие CRUD-эндпоинты
# ------------------

# Получить список ресторанов
@app.route('/get_restaurants', methods=['GET'])
def get_restaurants():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, address, rating, cuisine FROM restaurants;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    items = []
    for r in rows:
        items.append({
            'id':      r[0],
            'name':    r[1],
            'address': r[2],
            'rating':  r[3],
            'cuisine': r[4]
        })
    return Response(json.dumps({'restaurants': items}, ensure_ascii=False),
                    mimetype='application/json; charset=utf-8')


# Добавить отзыв о шефе
@app.route('/add_chef_review', methods=['POST'])
def add_chef_review():
    data = request.json
    chef_id = data.get('to_id')
    rating  = data.get('rating')
    comment = data.get('comment')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SET NAMES utf8mb4 COLLATE utf8mb4_general_ci")
    cursor.execute(
        "INSERT INTO chef_reviews (from_id, to_id, rating, comment, created_at) "
        "VALUES (%s,%s,%s,%s,NOW())",
        (data.get('from_id'), chef_id, rating, comment)
    )
    conn.commit()
    rid = cursor.lastrowid
    cursor.close()
    conn.close()
    return jsonify({"message": "Отзыв добавлен", "review_id": rid})


# Получить отзывы о шефах
@app.route('/get_chef_reviews', methods=['GET'])
def get_chef_reviews():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SET NAMES utf8mb4 COLLATE utf8mb4_general_ci")
    cursor.execute("SELECT id, from_id, to_id, rating, comment FROM chef_reviews;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    reviews = [{
        'id':       r[0],
        'from_id':  r[1],
        'to_id':    r[2],
        'rating':   r[3],
        'comment':  r[4]
    } for r in rows]
    return jsonify({'chef_reviews': reviews})


# Добавить скрапнутый отзыв
@app.route('/add_scraped_review', methods=['POST'])
def add_scraped_review():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scraped_reviews (restaurant_id, source, rating, review_text, review_date, created_at) "
        "VALUES (%s,%s,%s,%s,%s,NOW())",
        (data.get('restaurant_id'), data.get('source'),
         data.get('rating'), data.get('review_text'), data.get('review_date'))
    )
    conn.commit()
    rid = cursor.lastrowid
    cursor.close()
    conn.close()
    return jsonify({"message": "Скрапнутый отзыв добавлен", "review_id": rid})


# Получить скрапнутые отзывы
@app.route('/get_scraped_reviews', methods=['GET'])
def get_scraped_reviews():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, restaurant_id, source, rating, review_text, review_date FROM scraped_reviews;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    reviews = [{
        'id':            r[0],
        'restaurant_id': r[1],
        'source':        r[2],
        'rating':        r[3],
        'review_text':   r[4],
        'review_date':   str(r[5])
    } for r in rows]
    return jsonify({'scraped_reviews': reviews})


# Наличие ping для проверки работоспособности
@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


if __name__ == '__main__':
    app.run(debug=True)
