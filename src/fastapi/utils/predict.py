import joblib
import os
import joblib
from fastapi import HTTPException

def predict_text(model_name:str, texts:str):
    # vectorize data
    vectorizer = joblib.load("data/models/vectorizer.joblib")
    # load model
    model = joblib.load(f"data/models/{model_name}.joblib")


    text_pred = vectorizer.transform(texts)

    # get pred
    pred = model.predict(text_pred)
    
    return pred
 