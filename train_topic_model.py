# Тематическое моделирование
#Загружает тексты отзывов, создаёт gensim.Dictionary + BOW, обучает LdaModel.
#Сохраняет модель и словарь в models/.
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

# 1) .env → БД
load_dotenv()
DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL)

# 2) читаем все тексты
df = pd.read_sql("SELECT review_text FROM scraped_reviews", engine)
texts = [t.lower().split() for t in df["review_text"].dropna()]

# 3) готовим словарь + корпус
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 4) обучаем LDA (например, 10 тем)
lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    passes=10,
    random_state=42
)

# 5) сохраняем
os.makedirs("models", exist_ok=True)
dictionary.save("models/lda_dictionary.gensim")
lda.save("models/lda_model.gensim")
print("Сохранено: models/lda_dictionary.gensim, models/lda_model.gensim")
