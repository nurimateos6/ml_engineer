from sklearn.linear_model import LinearRegression
from data.load_data import load_data
from data.clean_data import clean_data
from data.split_data import split_data
from model.validation import validation_results
import pickle


def run_model():
    """ Function that takes care of data cleaning, splitting of the initial dataset into train and test, training and
    validation of the model, as well as displaying the results. """
    X, y = load_data(to_numpy=True)
    clean_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)
    lr = LR()
    lr.run_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    validation_results(X_train, y_train, y_test, y_pred, lr.model)


class LR:
    """ LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of
    squares between the observed targets in the dataset, and the targets predicted by the linear approximation """

    def __init__(self):
        self.model = LinearRegression()
        self.params = self.model.get_params()

    def set_params(self, **params):
        """ Set the parameters of this estimator """
        self.model.set_params(params)

    def run_model(self, X_train, y_train):
        """ Fit linear model """
        self.model.fit(X_train, y_train)
        self.save_model()

    def save_model(self):
        filename = 'model.pkl'
        pickle.dump(self.model, open(filename, 'wb'))

    def predict(self, X_test):
        """ Predict using the linear model """
        return self.model.predict(X_test)
