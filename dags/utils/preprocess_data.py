import nltk
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

import pandas as pd


def preprocess_data():
    # nltk utils
    nltk.download("omw-1.4")
    nltk.download("wordnet")
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    

    # read data
    spark = SparkSession.builder.getOrCreate() 
    df = pd.read_csv("data/jigsaw-toxic-comment-train.csv")
    df = spark.createDataFrame(df)

    
    # drop rows with nans
    df = df.na.drop()
    # preprocessing functions
    to_lower_case = udf(lambda s: s.lower().replace("\n", " "), StringType())
    tokenization = udf(lambda x: tokenizer.tokenize(x),)
    lemmatization = udf(lambda x: " ".join([lemmatizer.lemmatize(w) for w in x]))
    # apply preprocessing
    df = df.withColumn('comment_text', to_lower_case("comment_text"))
    df = df.withColumn('comment_text', tokenization("comment_text"))
    df = df.withColumn('comment_text', lemmatization("comment_text"))
    # remove epmty rows
    df = df.filter(df.comment_text != "")

    # save to csv
    df.toPandas().to_csv('data/preprocessed_data.csv')
