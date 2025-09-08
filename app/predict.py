from app.model import load_model
import numpy as np

model = load_model()

def predict_species(features: list):
    prediction = model.predict([features])
    return int(prediction[0])
