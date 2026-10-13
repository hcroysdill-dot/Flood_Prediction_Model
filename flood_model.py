from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def learn_rand_forest(train_X, train_y, random_state=42):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(train_X, train_y)
    return model


def learn_logistic_regression(train_X, train_y):
    model = LogisticRegression()
    model.fit(train_X, train_y)
    return model


def learn_SVC(train_X, train_y):
    model = SVC()
    model.fit(train_X, train_y)
    return model


def learn_MLP(train_X, train_y, random_state=42):
    # The MLP can give inconsistent results between runs;
    # random_state keeps it repeatable.
    model = MLPClassifier(max_iter=1000, random_state=random_state)
    model.fit(train_X, train_y)
    return model


def make_learners(random_state=42):
    """Return {model name: function(train_X, train_y) -> trained model}."""
    return {
        "Random Forest": partial(learn_rand_forest, random_state=random_state),
        "Logistic Regression": learn_logistic_regression,
        "SVC": learn_SVC,
        "MLP": partial(learn_MLP, random_state=random_state),
    }
