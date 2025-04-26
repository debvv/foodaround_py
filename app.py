import os
import json
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd

# Загрузка переменных окружения из .env
load_dotenv()
DB_URL = os.getenv("DB_URL")  # например "mysql+pymysql://root:пароль@localhost/foodaround_db"
if not DB_URL:
    raise RuntimeError("Переменная окружения DB_URL не задана")

# Создаём SQLAlchemy-движок для pandas и других видов запросов
engine = create_engine(DB_URL, pool_pre_ping=True)

app = Flask(__name__)

# Оригинальный MySQL-коннектор для cursor-based операций
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", "foodaround_db"),
        charset='utf8mb4',
        use_unicode=True,
        use_pure=True
    )

@app.route("/ping")
def ping():
    return "pong"

@app.route('/get_restaurants', methods=['GET'])
def get_restaurants():
    # Читаем данные через pandas + SQLAlchemy
    df = pd.read_sql(
        "SELECT id, name, address, rating, cuisine FROM restaurants;",
        con=engine
    )
    restaurants = df.to_dict(orient='records')
    return Response(
        json.dumps({"restaurants": restaurants}, ensure_ascii=False),
        mimetype='application/json; charset=utf-8'
    )

@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    data = request.json
    # TODO: загрузить models/demand_model.pkl и restaurant_encoder.pkl, сделать реальный прогноз
    return jsonify({"prediction": "Пример предсказания спроса"})

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json.get("user_id")
    # TODO: загрузить CF- и CB-модели и вернуть реальные рекомендации
    return jsonify({"recommendations": ["Ресторан A", "Ресторан B"]})

@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    review_text = request.json.get("review_text")
    # TODO: загрузить sentiment_model.pkl и вернуть тональность
    return jsonify({"sentiment": "положительный"})

@app.route('/add_chef_review', methods=['POST'])
def add_chef_review():
    data = request.json
    chef_name = data.get("chef_name")
    rating = data.get("rating")
    comment = data.get("comment")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SET NAMES utf8mb4 COLLATE utf8mb4_general_ci")
    cursor.execute(
        "INSERT INTO chef_reviews (chef_name, rating, comment, created_at) "
        "VALUES (%s, %s, %s, NOW())",
        (chef_name, rating, comment)
    )
    conn.commit()
    review_id = cursor.lastrowid
    cursor.close()
    conn.close()

    return jsonify({"message": "Отзыв добавлен!", "review_id": review_id})

@app.route('/get_chef_reviews', methods=['GET'])
def get_chef_reviews():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SET NAMES utf8mb4 COLLATE utf8mb4_general_ci")
    cursor.execute("SELECT id, chef_name, rating, comment, created_at FROM chef_reviews;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    reviews = []
    for _id, chef_name, rating, comment, created_at in rows:
        reviews.append({
            "id": _id,
            "chef_name": chef_name,
            "rating": rating,
            "comment": comment,
            "created_at": str(created_at)
        })
    return jsonify({"chef_reviews": reviews})

@app.route('/add_scraped_review', methods=['POST'])
def add_scraped_review():
    data = request.json
    restaurant_name = data.get("restaurant_name")
    source = data.get("source")
    rating = data.get("rating")
    review_text = data.get("review_text")
    review_date = data.get("review_date")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scraped_reviews "
        "(restaurant_name, source, rating, review_text, review_date, created_at) "
        "VALUES (%s, %s, %s, %s, %s, NOW())",
        (restaurant_name, source, rating, review_text, review_date)
    )
    conn.commit()
    review_id = cursor.lastrowid
    cursor.close()
    conn.close()

    return jsonify({"message": "Скрапнутый отзыв добавлен!", "review_id": review_id})

@app.route('/get_scraped_reviews', methods=['GET'])
def get_scraped_reviews():
    df = pd.read_sql(
        "SELECT id, restaurant_name, source, rating, review_text, review_date FROM scraped_reviews;",
        con=engine,
        parse_dates=['review_date']
    )
    reviews = df.to_dict(orient='records')
    # Преобразуем даты в строку
    for r in reviews:
        r['review_date'] = str(r['review_date'])
    return Response(
        json.dumps({"scraped_reviews": reviews}, ensure_ascii=False),
        mimetype='application/json; charset=utf-8'
    )

@app.route('/update_data', methods=['POST'])
def update_data():
    # TODO: по запросу обновить данные (скрап, retrain, etc.)
    return jsonify({"status": "Данные обновлены"})

if __name__ == '__main__':
    app.run(debug=True)
