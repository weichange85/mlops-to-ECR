from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_species

app = FastAPI(title="Iris Classifier")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisFeatures):
    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]
    species = predict_species(features)
    return {"predicted_class": species}
