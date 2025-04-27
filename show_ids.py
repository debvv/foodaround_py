# show_ids.py
import joblib

encoder = joblib.load("models/restaurant_encoder.pkl")
print("Known restaurant_id:", encoder.classes_)
