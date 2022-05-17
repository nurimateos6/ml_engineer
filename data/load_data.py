from sklearn.datasets import load_boston
import pandas as pd
import warnings
import os.path

PATH = 'src_data'
FILE_NAME = 'boston_data.csv'


def is_path():
    if not os.path.isdir(PATH):
        os.mkdir(PATH)


def is_file():
    is_path()
    return os.path.isfile(os.path.join(PATH, FILE_NAME))


def get_data():
    """ Load Boston dataset: : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html"""
    if not is_file():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            boston_data = load_boston()
            X, y = boston_data.data, boston_data.target
            colnames = boston_data.feature_names
            X_df = pd.DataFrame(X, columns=colnames)
            X_df['y'] = y
            X_df.to_csv(os.path.join(PATH, FILE_NAME), index=False)


def load_data(to_numpy=False):
    df = pd.read_csv(os.path.join(PATH, FILE_NAME))
    if to_numpy:
        X = df.iloc[:, :-1].to_numpy()
        y = df['y'].to_numpy()
        return X, y
    else:
        return df
