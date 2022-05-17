from sklearn.datasets import load_boston
import pandas as pd
import warnings
import os

PATH = 'src_data'
FILE_NAME = 'boston_data.csv'


def is_path():
    if not os.path.isdir(PATH):
        os.mkdir(PATH)


def is_file():
    is_path()
    return os.path.isfile(os.path.join(PATH, FILE_NAME))


def load_data(to_numpy=False):
    is_file()
    df = pd.read_csv(os.path.join(PATH, FILE_NAME))
    if to_numpy:
        X = df.iloc[:, :-1].to_numpy()
        y = df['y'].to_numpy()
        return X, y
    else:
        return df


def __load_boston():
    """ Load Boston dataset: : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        boston_data = load_boston()
        X, y = boston_data.data, boston_data.target
        colnames = boston_data.feature_names
        X_df = pd.DataFrame(X, columns=colnames)
        X_df['y'] = y
        is_file()
        X_df.to_csv(os.path.join(PATH, FILE_NAME), index=False)
        return X_df


def test_dataframe():
    """ Simple test to check that two dataframes are equals """
    assert __load_boston().compare(load_data()).empty


def test_dataframe_columns():
    """ Simple test to check that two dataframes have the same columns """
    assert all(__load_boston().columns == load_data().columns)
