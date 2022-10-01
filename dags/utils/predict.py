import json

import joblib
import pandas as pd


def predict():
    # load model
    model = joblib.load("data/model.joblib")

    # chose random row
    df = pd.read_csv("data/preprocessed_data.csv")
    df_pred = df.sample(1)[["comment_text"]]

    # vectorize data
    vectorizer = joblib.load("data/vectorizer.joblib")
    df_sample = vectorizer.transform(df_pred["comment_text"])

    # get pred
    pred = model.predict(df_sample)
    pred = {"text": df_pred["comment_text"].item(), "pred":pred[0]}
    with open("data/predict.json", "w") as f:
        json.dump(pred, f)
 
