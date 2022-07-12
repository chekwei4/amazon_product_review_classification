"""This module provides definitions of predictive models to be
trained."""

from sklearn.linear_model import LogisticRegression
# import sklearn

def logistic_regression_model(args):
    return LogisticRegression()
    # return sklearn.linear_model.base.LogisticRegression