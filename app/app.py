import pickle
import xgboost as xgb
import uvicorn

from fastapi import FastAPI
# from flask import Flask, request, jsonify 
from typing import Dict, Any
from src.predict import predict

app = FastAPI(title = "subscribe")

with open('../model/bank-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


@app.post("/predict")
def predict_end(data: Dict[str, Any]):
    score = predict(data)
    if score == 0:
        result = 'Not subscribed'
    else:
        result = 'Subscribed'
    return {
        f"This user is {result}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)