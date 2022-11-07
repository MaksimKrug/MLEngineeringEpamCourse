import json

import joblib
import pandas as pd


def predict(**kwargs):
    # load model
    model = joblib.load("data/model.joblib")

    # get text
    try:
        text = kwargs['params']["text"]
    except:
        text = "default text"

    # vectorize data
    vectorizer = joblib.load("data/vectorizer.joblib")
    text_pred = vectorizer.transform([text])

    # get pred
    pred = model.predict(text_pred)
    print(model.predict_proba(text_pred))
    pred_save = {"text": text, "pred": str(pred[0])}
    with open("data/predict.json", "w") as f:
        json.dump(pred_save, f)
 
