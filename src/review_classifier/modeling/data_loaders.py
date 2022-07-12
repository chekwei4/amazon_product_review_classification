"""This module contains functions that assist in loading datasets
for the model training pipeline."""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import pickle

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

from . import utils

logging.getLogger(__name__)

def load_datasets(current_working_dir, args):
    logging.info("current_working_dir " + current_working_dir)
    data_path = os.path.join(current_working_dir, args["train"]["data_path"])
    logging.info(f"showing data_path...{data_path}")
    df = pd.read_csv(data_path)
    logging.info(f"successfully read from clean_review.csv...{df.shape}")
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info("successfully completed train test split...")
    logging.info(f'Original dataset shape : {Counter(y_train)}')
    train_tfidf_matrix, tfidf_vectorizer = create_train_matrix_vectorizer(X_train)

    utils.export_vectorizer(tfidf_vectorizer)

    test_tfidf_matrix = create_test_matrix(tfidf_vectorizer, X_test)
    logging.info("successfully created tfidf matrix...")

    logging.info("applying SMOTE...")

    X_res, y_res = apply_smote(train_tfidf_matrix, y_train)
    logging.info("done with SMOTE...")

    return X_res, test_tfidf_matrix, y_res, y_test


def get_X_y(df):
    y = df.sentiment
    X = df.lemmatized_review
    y = np.array(list(map(lambda x: 1 if x == "pos" else 0, y)))
    return X, y


def tokenize_lines(review):
    return nltk.word_tokenize(review)

def tokenize(review):
    review = tokenize_lines(review)
    return review

def create_train_matrix_vectorizer(X_train):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    train_tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)
    return train_tfidf_matrix, tfidf_vectorizer


def create_test_matrix(vectorizer, X_test):
    test_tfidf_matrix = vectorizer.transform(X_test)
    return test_tfidf_matrix

def apply_smote(tfidf_matrix, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(tfidf_matrix, y_train)
    return X_res, y_res