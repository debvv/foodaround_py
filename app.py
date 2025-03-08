import json
from flask import Flask, request, jsonify
from flask import Response
import psycopg2




app = Flask(__name__)

# Заглушка для предсказания спроса
@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    data = request.json  # Получаем входные данные
    # TODO: Добавить обработку данных и предсказание
    return jsonify({"prediction": "Пример предсказания спроса"})



# Заглушка для системы рекомендаций
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json.get("user_id")
    # TODO: Добавить алгоритм рекомендаций
    return jsonify({"recommendations": ["Ресторан 1", "Ресторан 2"]})



# Заглушка для анализа отзывов
@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    review_text = request.json.get("review_text")
    # TODO: Добавить NLP-модель для анализа отзывов
    return jsonify({"sentiment": "положительный"})



# Заглушка для получения ресторанов
@app.route('/get_restaurants', methods=['GET'])
def get_restaurants():
    # TODO: Добавить базу данных с ресторанами
    data = {"restaurants": ["Ресторан A", "Ресторан B"]}
    return Response(json.dumps(data, ensure_ascii=False), mimetype='application/json')


    #return jsonify({"restaurants": ["Ресторан A", "Ресторан B"]}), 200, {
    #        'Content-Type': 'application/json; charset=utf-8'}




# Заглушка для обновления данных
@app.route('/update_data', methods=['POST'])
def update_data():
    # TODO: Интегрировать парсер и обновление базы
    return jsonify({"status": "Данные обновлены"})



#Добавляем эндпоинт для добавления отзыва о кулинаре
@app.route('/add_chef_review', methods=['POST'])
def add_chef_review():
    data = request.json
    chef_name = data.get("chef_name")
    rating = data.get("rating")
    comment = data.get("comment")

    conn = psycopg2.connect(
        dbname="foodaround_db",
        user="postgres",
        password="ТВОЙ_ПАРОЛЬ",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO chef_reviews (chef_name, rating, comment) VALUES (%s, %s, %s) RETURNING id;",
        (chef_name, rating, comment)
    )
    review_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Отзыв о кулинаре добавлен!", "review_id": review_id})



# Добавляем эндпоинт для получения отзывов о кулинарах
@app.route('/get_chef_reviews', methods=['GET'])
def get_chef_reviews():
    conn = psycopg2.connect(
        dbname="foodaround_db",
        user="postgres",
        password="ТВОЙ_ПАРОЛЬ",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT id, chef_name, rating, comment FROM chef_reviews;")
    reviews = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify({"chef_reviews": reviews})


# testing curl -X POST http://127.0.0.1:5000/add_chef_review -H "Content-Type: application/json" -d '{"chef_name": "Гордон Рамзи", "rating": 5, "comment": "Отличный повар!"}'


if __name__ == '__main__':
    app.run(debug=True)
