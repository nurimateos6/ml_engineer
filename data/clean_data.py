import numpy as np


def check_data(X: np.ndarray, y: np.ndarray):
    """ Check that the dimension of the features and the target are equal"""
    return True if X.shape[0] == y.shape[0] else False


def nan_to_value(x: np.ndarray, value: float):
    """ replaces null values by a given value """
    np.nan_to_num(x, value)


def check_nulls(x: np.ndarray):
    """ checks if an array has null values """
    if np.isnan(x).any():
        nan_to_value(x, np.mean(x))


def replace_nulls(X: np.ndarray, y: np.ndarray):
    """ Replace all null values in the list with the subsets of train and test"""
    l = [X, y]
    [check_nulls(array) for array in l]


def clean_data(X: np.ndarray, y: np.ndarray):
    if check_data(X, y):
        replace_nulls(X, y)
    else:
        "Check shape of the input data"

