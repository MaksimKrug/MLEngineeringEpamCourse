import nltk
import pandas as pd

# nltk utils
nltk.download("omw-1.4")
nltk.download("wordnet")
lemmatizer = nltk.stem.WordNetLemmatizer()
tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")


def preprocess_data():
    # read data
    df = pd.read_csv("data/jigsaw-toxic-comment-train.csv", index_col=0)

    # Sample data
    df = df.sample(10000)

    # preprocess text
    df["comment_text"] = df["comment_text"].apply(
        lambda x: x.lower().replace("\n", " ")
    )
    df["comment_text"] = df["comment_text"].apply(lambda x: tokenizer.tokenize(x))
    df["comment_text"] = df["comment_text"].apply(
        lambda x: " ".join([lemmatizer.lemmatize(w) for w in x])
    )

    # remove empty texts
    df = df.loc[df["comment_text"].notna()]
    df = df.loc[df["comment_text"] != ""]

    # save preprocessed data
    df.to_csv("data/preprocessed_data.csv", index=False)