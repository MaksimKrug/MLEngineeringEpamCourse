import json
import joblib

import pandas as pd
import seaborn as sns
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
joblib.dump(vectorizer, "vectorizer.joblib")

# preprocess target
target_columns = [
    "toxic",
]
y_train = train_data[target_columns].values
y_test = test_data[target_columns].values

# train model
model = RandomForestClassifier(random_state=42, max_depth=10)
model.fit(X_train, y_train.ravel())

# metrics
y_pred = model.predict(X_test)
acc, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc, "f1_score": f1}, f)

# get the most important words
all_words = vectorizer.get_feature_names_out()
words_importnaces = model.feature_importances_
the_most_important_words = sorted(
    list(zip(all_words, words_importnaces)), key=lambda x: x[1], reverse=True
)
temp_data = pd.DataFrame(the_most_important_words, columns=["Words", "Importance"])
fig = sns.barplot(x="Words", y="Importance", data=temp_data[:10])
fig.get_figure().savefig("feature_importances.png")

# save model
joblib.dump(model, "model.joblib")
