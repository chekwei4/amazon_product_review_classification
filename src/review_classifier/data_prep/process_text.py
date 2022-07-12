"""This module contains functions for cleaning text data."""

import logging
import re
import string

import gzip
import pandas as pd

import datetime

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

logger = logging.getLogger(__name__)

FILE_NAME = "reviews_Clothing_Shoes_and_Jewelry_5.json.gz"

def process_file(file_path):
    # logger.debug("Reading text file: {}".format(file_path))
    logging.info(f"Printing file path from process_text.py...{file_path}")
    logging.info(f"file name is... {FILE_NAME}")
    #to run locally, please comment out below line
    file_path = file_path + "/" + FILE_NAME
    #to run locally, please comment out above line
    df = get_dataframe(file_path)
    df = drop_features(df)
    df = create_review_time(df)
    df = create_review(df)
    df = create_sentiment(df)
    df["reviewText_len"] = df["review"].astype(str).apply(len)
    df = drop_review_len_zero(df)
    df["review_no_punct"] = df["review"].apply(remove_punctuation)
    # df = df[:100]
    logging.info(f"process_text.py shape before lemmatize:{df.shape}")
    logging.info("running lemmatizing...1aag")
    # print(df.columns)
    # print(df.head(3))
    logging.info(f"before lemmatizing dot head...{df.head(3)}")
    df["lemmatized_review"] = df["review_no_punct"].apply(lemmatize_review)
    logging.info(f"after lemmatizing dot head...{df.head(3)}")
    logging.info(f"process_text.py shape after lemmatize:{df.shape}")
    logging.info("before lemma text below...")
    logging.info(df[["review_no_punct", "lemmatized_review"]].loc[0][0])
    logging.info("after lemma text below...")
    logging.info(df[["review_no_punct", "lemmatized_review"]].loc[0][1])
    logging.info("process_file.py ends here...")
    return df


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def get_dataframe(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def drop_features(df):
    return df.drop(["reviewerID", "reviewerName", "helpful", "unixReviewTime"], axis=1)


def create_review_time(df):
    # Splitting the date
    temp_dt = df["reviewTime"].str.split(",", n=1, expand=True)
    # adding date to the main dataset
    df["date"] = temp_dt[0]
    # adding year to the main dataset
    df["year"] = temp_dt[1]
    temp_dt = df["date"].str.split(" ", n=1, expand=True)
    # adding month to the main dataset
    df["month"] = temp_dt[0]
    # adding day to the main dataset
    df["day"] = temp_dt[1]
    df = df.drop(["date", "reviewTime"], axis=1)
    df["reviewTime"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.drop(["year", "month", "day"], axis=1)
    return df


def create_review(df):
    df["review"] = df["reviewText"] + " " + df["summary"]
    df = df.drop(["reviewText", "summary"], axis=1)
    return df


def create_bins(rating):
    if rating >= 4.0:
        val = "pos"
    else:
        val = "neg"
    return val


def create_sentiment(df):
    df["sentiment"] = df["overall"].apply(create_bins)
    return df


def drop_review_len_zero(df):
    return df.drop(df[df["reviewText_len"] == 0].index)


def remove_punctuation(review):
    no_punct = [words.lower() for words in review if words not in string.punctuation]
    review_wo_punct = "".join(no_punct)
    return review_wo_punct


def lemmatize_review(review):  
    lemmatizer = nltk.WordNetLemmatizer()
    tokenized_lines = nltk.word_tokenize(review)
    for i in range(len(tokenized_lines)):
        lammatize_words = [
            lemmatizer.lemmatize(word).lower()
            for word in tokenized_lines
            if not word.lower() in set(stopwords.words("english"))
        ]
        return " ".join(lammatize_words)





