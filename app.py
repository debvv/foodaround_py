from flask import Flask, request, jsonify, Response
import json
import mysql.connector

app = Flask(__name__)

# Подключение к базе данных MySQL


def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="foodaround_db"
    )


# Предсказание спроса (заглушка)
@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    data = request.json
    return jsonify({"prediction": "Пример предсказания спроса"})

# Система рекомендаций (заглушка)
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json.get("user_id")
    return jsonify({"recommendations": ["Ресторан 1", "Ресторан 2"]})


# Анализ отзывов (заглушка)
@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    review_text = request.json.get("review_text")
    return jsonify({"sentiment": "положительный"})


# Получение ресторанов (пока без данных из БД)
@app.route('/get_restaurants', methods=['GET'])
def get_restaurants():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, address, rating, cuisine FROM restaurants;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    restaurants = []
    for row in rows:
        restaurants.append({
            "id": row[0],
            "name": row[1],
            "address": row[2],
            "rating": row[3],
            "cuisine": row[4]
        })
    return jsonify({"restaurants": restaurants})


# Добавление поварского отзыва
@app.route('/add_chef_review', methods=['POST'])
def add_chef_review():
    data = request.json
    chef_name = data.get("chef_name")
    rating = data.get("rating")
    comment = data.get("comment")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chef_reviews (chef_name, rating, comment, created_at) VALUES (%s, %s, %s, NOW())",
        (chef_name, rating, comment)
    )
    conn.commit()
    review_id = cursor.lastrowid
    cursor.close()
    conn.close()

    return jsonify({"message": "Отзыв добавлен!", "review_id": review_id})


# Получение поварских отзывов
@app.route('/get_chef_reviews', methods=['GET'])
def get_chef_reviews():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, chef_name, rating, comment FROM chef_reviews;")
    reviews = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for r in reviews:
        result.append({"id": r[0], "chef_name": r[1], "rating": r[2], "comment": r[3]})

    return jsonify({"chef_reviews": result})


# Добавление скрапнутого отзыва
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
        "INSERT INTO scraped_reviews (restaurant_name, source, rating, review_text, review_date, created_at) "
        "VALUES (%s, %s, %s, %s, %s, NOW())",
        (restaurant_name, source, rating, review_text, review_date)
    )
    conn.commit()
    review_id = cursor.lastrowid
    cursor.close()
    conn.close()

    return jsonify({"message": "Скрапнутый отзыв добавлен!", "review_id": review_id})


# Получение скрапнутых отзывов
@app.route('/get_scraped_reviews', methods=['GET'])
def get_scraped_reviews():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, restaurant_name, source, rating, review_text, review_date FROM scraped_reviews;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    reviews = []
    for row in rows:
        reviews.append({
            "id": row[0],
            "restaurant_name": row[1],
            "source": row[2],
            "rating": row[3],
            "review_text": row[4],
            "review_date": str(row[5])
        })

    return jsonify({"scraped_reviews": reviews})


# Обновление данных (заглушка)
@app.route('/update_data', methods=['POST'])
def update_data():
    return jsonify({"status": "Данные обновлены"})


if __name__ == '__main__':
    app.run(debug=True)
