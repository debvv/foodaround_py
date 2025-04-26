# Тематическое моделирование
#Загружает тексты отзывов, создаёт gensim.Dictionary + BOW, обучает LdaModel.
#Сохраняет модель и словарь в models/.

import os
import pandas as pd
import gensim
from gensim.corpora import Dictionary
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL")

# 1. Загружаем тексты отзывов
engine = create_engine(DB_URL)
reviews = pd.read_sql("SELECT review_text FROM scraped_reviews", engine)

# 2. Предобработка: токенизация (можно добавить стоп-слова, лемматизацию)
texts = reviews['review_text'].fillna('').str.lower().str.split()

# 3. Словарь и корпус
dictionary = Dictionary(texts)
corpus     = [dictionary.doc2bow(text) for text in texts]

# 4. LDA-модель
lda = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    random_state=42,
    passes=5,
    alpha='auto'
)

# 5. Сохраняем
os.makedirs("models", exist_ok=True)
lda.save("models/lda_model.gensim")
dictionary.save("models/lda_dictionary.gensim")
print("LDA модель и словарь сохранены в models/")
