import joblib
import random

def load_model():
    return joblib.load("model.pkl")

def predict_duration(model, input_df):
    return model.predict(input_df)[0]

def assign_worker(duration):
    # Dummy worker logic based on duration
    return random.choice(["A", "B", "C"])





