from sklearn.model_selection import train_test_split
from data.clean_data import check_data
from numpy import ndarray


def check_correct_shape(train: tuple, test: tuple):
    """ checks that in the training and test subsets the features and the target are of the same dimension """
    return [check_data(X, y) for (X, y) in [train, test]]


def split_data(X: ndarray, y: ndarray):
    """ Split the data into training and testing subsets. The data is shuffled into a random orden when creating the
    subsets for remove any bias """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    results = check_correct_shape((X_train, y_train), (X_test, y_test))
    if any(results):
        return X_train, X_test, y_train, y_test
    else:
        print('Check if the division of subsets is correct')