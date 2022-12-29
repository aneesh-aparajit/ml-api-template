from typing import Optional
from fastapi import FastAPI
from ml import TextModel, BERTDataset, train_model
import numpy as np
from pydantic import BaseModel

app = FastAPI()


class SentimentPredict(BaseModel):
    text: str
    threshold: float = 0.5


MODEL = TextModel()
MODEL.load("model.bin", device="mps")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def fetch_predictions(sp: SentimentPredict):
    dataset = BERTDataset([sp.text], [-1])
    pred = float(list(MODEL.predict(dataset=dataset))[0][0][0])
    prediction = 1 / (1 + np.exp(pred))
    sentiment = None

    if prediction > sp.threshold:
        sentiment = "positive"
    else:
        sentiment = "negative"

    return {
        "positive": prediction,
        "negative": 1 - prediction,
        "sentence": sp.text,
        "threshold": sp.threshold,
        "sentiment": sentiment,
    }
