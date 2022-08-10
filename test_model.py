import json
import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

metrics_path = "metrics.json"
model_path = "model.joblib"
vectorizer_path = "vectorizer.joblib"

# files exists
def test_model_exists():
    assert os.path.exists(model_path), "Model file doesn't exists"
    assert os.path.exists(metrics_path), "Metrics file doesn't exists"
    assert os.path.exists(vectorizer_path), "Vectorizer (tfidf) file doesn't exists"


# load metrics and model
with open(metrics_path, "r") as f:
    metrics = json.load(f)
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# accuracy higher than threshold
def test_accuracy(th=0.7):
    assert metrics["accuracy"] >= th, "Accuracy is too low, we are sorry"

# check that model return some values
def test_model():
    random_text = np.array(["here is some random text", "and not random text"])
    predict = model.predict(vectorizer.transform(random_text))
    assert all([i == 0 or i == 1 for i in predict]), "Model not binary"