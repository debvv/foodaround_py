# seed_orders.py
# seed_orders.py
import mysql.connector
from datetime import datetime

# 1. подключаемся
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="foodaround_db",
    charset="utf8mb4",
    use_unicode=True,
    use_pure=True
)
cur = conn.cursor()

# 2. выбираем любой существующий unique_id из users
cur.execute("SELECT unique_id FROM users LIMIT 1")
row = cur.fetchone()
if not row:
    raise RuntimeError("В таблице users нет ни одной записи — сначала добавьте хотя бы одного пользователя!")
from_id = row[0]
print(f"Будем считать, что заказы делает пользователь с unique_id = {from_id!r}")

# 3. Сеем по 5 заказов для ресторанов 1,2,3
for rid in (1, 2, 3):
    for i in range(5):
        dt = datetime(2025, 4, 26, 10 + i, 0, 0)
        cur.execute(
            """
            INSERT INTO orders
                (restaurant_id, time_date, name, from_id,
                 product, description, count, email, address, accepted)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                rid,
                dt,
                f"TestUser{rid}",  # любое имя
                from_id,
                "TestDish",
                "TestDesc",
                1,
                "test@example.com",
                "Some Address",
                rid  # допустим, «принял» тот же ресторан_id
            )
        )

conn.commit()
cur.close()
conn.close()

print("✓ Засеяно тестовых записей в таблицу orders.")

#Заполнить таблицу orders реальными restaurant_id
# в схеме orders нет столбца restaurant_id, поэтому все restaurant_id при агрегации оказывались NULL.
#  добавим в неё этот столбец и заси́дим несколько тестовых записей