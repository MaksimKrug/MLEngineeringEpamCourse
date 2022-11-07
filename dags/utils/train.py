import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def train():
    """
    DATA
    """
    # load data
    df = pd.read_csv("data/preprocessed_data.csv")
    # split
    train_data, test_data = train_test_split(df, random_state=42)
    # TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data["comment_text"])
    X_test = vectorizer.transform(test_data["comment_text"])
    joblib.dump(vectorizer, "data/vectorizer.joblib")
    # preprocess target
    target_columns = [
        "toxic",
    ]
    y_train = train_data[target_columns].values
    y_test = test_data[target_columns].values

    """
    TRAIN MODEL
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train.ravel())
    
    # metrics
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
    }
    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f)
    # save model
    joblib.dump(model, "data/model.joblib")
