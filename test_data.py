"""
There are tests for data
"""
import os

import pandas as pd

# utils
train_path = "data/jigsaw-toxic-comment-train.csv"
processed_path = "data/preprocessed_data.csv"

# files exists
def test_file_exists():
    assert os.path.exists(train_path), "Train file doesn't exists"
    assert os.path.exists(processed_path), "Preprocessed file doesn't exists"


# read data
train_data = pd.read_csv(train_path, index_col=0)
preprocessed_data = pd.read_csv(processed_path)

# check dataframes not empty
def test_data_not_empty():
    assert train_data.size != 0, "Train data is empty"
    assert preprocessed_data.size != 0, "Preprocessed data is empty"


# label values
def test_labels_values():
    label_columns = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    assert all(train_data[label_columns].nunique() == 2), "Train labels are not binary"
    assert all(
        train_data[label_columns].nunique() == 2
    ), "Preprocessed labels are not binary"


# test text column text
def test_text_column_is_string():
    assert (
        train_data["comment_text"].str.contains("a").any()
    ), "Train text column is not a string"
    assert (
        preprocessed_data["comment_text"].str.contains("a").any()
    ), "Preprocessed text column is not a string"


# test preprocessed is lower case
def test_lower_case():
    assert all(
        preprocessed_data["comment_text"].str.lower()
        == preprocessed_data["comment_text"]
    ), "Preprocessed data not in lower case"


# test NaNs
def test_nans():
    assert all(preprocessed_data.isna().sum() == 0), "Preprocessed data contain NaNs"


# test empty texts
def test_empty_texts():
    assert (
        preprocessed_data[preprocessed_data["comment_text"] == ""].size == 0
    ), "Preprocessed data contain empty texts"
    assert (
        preprocessed_data[preprocessed_data["comment_text"] == " "].size == 0
    ), "Preprocessed data contain empty texts"


# test toxic label distribution
def test_toxic_distribution():
    assert (
        0.8 <= preprocessed_data["toxic"].value_counts(normalize=True).loc[0] <= 0.95
    ), "0 value distribution"
    assert (
        0.05 <= preprocessed_data["toxic"].value_counts(normalize=True).loc[1] <= 0.15
    ), "1 value distribution"
