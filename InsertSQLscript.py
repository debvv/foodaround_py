
import random
from datetime import date, timedelta

restaurants = ["McDonald''s","Madam Wong","Grill House","Zaxi Fun & Finest",
               "Buffalo Steak House","Test Resto","Demo Cafe"]
sources     = ["TripAdvisor","Google","Yelp","Facebook","LocalBlog"]
reviews     = [
    "Очень вкусно, но немного долгий сервис.",
    "Хорошее место, но цены слегка завышены.",
    "Отличный гриль и атмосферная подача блюд!",
    "Вкусно, но пришлось ждать заказ 30 минут.",
    "Хороший кофе и десерты, но обслуживание среднее.",
    "Ужасный сервис и холодная еда.",
    "Не рекомендую — долго приносили заказ.",
    "Приятная атмосфера, но порции слишком маленькие.",
    "Всё понравилось, вернусь ещё раз!",
    "Средний вкус, но хорошее соотношение цена/качество."
]
start = date(2025,4,1)
end   = date(2025,4,26)
for i in range(1,201):
    rid    = (i-1)%7
    src    = sources[(i-1)%5]
    rating = round(1.0 + ((i-1)%40)*0.1,1)
    text   = reviews[(i-1)%10]
    day    = ((i-1)%26)+1
    rdate  = f"2025-04-{day:02d}"
    print(
        "INSERT INTO `scraped_reviews` "
        "(`restaurant_name`,`source`,`rating`,`review_text`,`review_date`,`restaurant_id`) VALUES "
        f"('{restaurants[rid]}','{src}',{rating},'{text}','{rdate}',{rid+1});"
    )

