from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from model.cross_validation import get_cross_val_scores
from sklearn.linear_model import LogisticRegression
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt


def mae(y_true: ndarray, y_pred: ndarray):
    """ Mean absolute error regression loss """
    return mean_absolute_error(y_true, y_pred)


def mse(y_true: ndarray, y_pred: ndarray):
    """ Mean squared error regression loss """
    return mean_squared_error(y_true, y_pred)


def r_score(y_true: ndarray, y_pred: ndarray):
    """ r2 (coefficient of determination) regression score function. Best possible score is 1.0 and it can be
    negative (because the model can be arbitrarily worse) """
    return r2_score(y_true, y_pred)


def cross_validation(model: LogisticRegression, X_train: ndarray, y_train: ndarray):
    """ Evaluate a score by cross-validation """
    mean, std = get_cross_val_scores(model, X_train, y_train, cv=5)
    return mean, std


def df_results(y_true: ndarray, y_pred: ndarray):
    """ creates a dataframe with the model results and the actual values """
    return pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})


def plot_results(y_true: ndarray, y_pred: ndarray):
    """ Plot the results of the predictions """
    df_res = df_results(y_true, y_pred)
    plt.plot(df_res)
    plt.plot()
    plt.suptitle('Comparison of model results with actual values')
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.savefig('model_results.png')


def validation_results(X_train: ndarray, y_train: ndarray, y_test: ndarray, y_pred: ndarray, model: LogisticRegression):
    """ Function to validate model results """
    cross_validation(model, X_train, y_train)
    mae_error = mae(y_test, y_pred)
    mse_error = mse(y_test, y_pred)
    r2score = r_score(y_test, y_pred)
    with open('validation_metrics.txt', 'w+') as f:
        f.write(f'MAE: {mae_error}, MSE: {mse_error} and R2_score: {r2score}')
    plot_results(y_test, y_pred)
