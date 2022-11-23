import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import mlflow


def train(model_name: str):
    """
    Data preprocessing
    """
    # load data
    df = pd.read_csv("data/preprocessed_data.csv")
    # split
    train_data, test_data = train_test_split(df, random_state=42)
    # TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data["comment_text"])
    X_test = vectorizer.transform(test_data["comment_text"])
    joblib.dump(vectorizer, "data/models/vectorizer.joblib")
    # preprocess target
    target_columns = [
        "toxic",
    ]
    y_train = train_data[target_columns].values
    y_test = test_data[target_columns].values

    """
    Model training
    """
    if model_name == "RandomForest":
        model = RandomForestClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    elif model_name == "SVC":
        model = SVC()
    # train model
    model.fit(X_train, y_train.ravel())

    # metrics
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
    }
    # mlflow log
    mlflow.set_tracking_uri(os.environ["MLFLOW_IP"])
    with mlflow.start_run() as mlrun:
        mlflow.set_tag("Model Name", model_name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)

    # save to joblib
    joblib.dump(model, f"data/models/{model_name}.joblib")
