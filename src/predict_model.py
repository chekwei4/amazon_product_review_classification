from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import os
import logging
import nltk

def run_predict(model, vectorizer, inf_review_text):
    tfidf_matrix = create_matrix_vectorizer(vectorizer, inf_review_text)
    logging.info("Evaluating the model...")
    logging.info("Model prediction done...")
    return model.predict(tfidf_matrix), model.predict_proba(tfidf_matrix)  

def create_matrix_vectorizer(vectorizer, inf_review_text):
    # tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfidf_matrix = vectorizer.transform(inf_review_text)
    return tfidf_matrix

def tokenize(review):
    review = tokenize_lines(review)
    return review

def tokenize_lines(review):
    return nltk.word_tokenize(review)
