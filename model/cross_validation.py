import pandas as pd
import sklearn.linear_model
from sklearn.model_selection import cross_val_score
import numpy as np


def get_cross_val_scores(model: sklearn.linear_model.LinearRegression, X_train: pd.DataFrame, y_train: pd.DataFrame, cv: int):
    """ Function to evaluate a score by cross-validation. Returns the mean and std for the array of scores of the
    estimator for each run of the cross validation """
    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             cv=cv,
                             scoring='r2')
    mean = np.mean(scores)
    std = np.std(scores)
    print(f'Mean: {mean} and Std: {std}')
    return mean, std
