import nltk
import pandas as pd

# nltk utils
nltk.download("omw-1.4")
nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()
tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")

# read data
df = pd.read_csv("data/jigsaw-toxic-comment-train.csv", index_col=0)
df = df.sample(1000, random_state=42)

# preprocess text
df["comment_text"] = df["comment_text"].apply(lambda x: x.lower().replace("\n", " "))
df["comment_text"] = df["comment_text"].apply(lambda x: tokenizer.tokenize(x))
df["comment_text"] = df["comment_text"].apply(
    lambda x: " ".join([lemmatizer.lemmatize(w) for w in x])
)

# save preprocessed data
df.to_csv("data/preprocessed_data.csv", index=False)
