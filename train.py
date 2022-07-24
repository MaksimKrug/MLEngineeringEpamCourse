import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv("data/preprocessed_data.csv")

# split
train_data, test_data = train_test_split(df, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data["comment_text"])
X_test = vectorizer.transform(test_data["comment_text"])

# preprocess target
target_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
y_train = train_data[target_columns].values
y_test = test_data[target_columns].values

# train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# metrics
y_pred = model.predict(X_test)
acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="micro")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc, "f1_score": f1}, f)
 