from sklearn.datasets import make_classification

from flood_model import make_learners

MODEL_NAMES = {"Random Forest", "Logistic Regression", "SVC", "MLP"}


def make_easy_data():
    return make_classification(
        n_samples=200, n_features=3, n_informative=3, n_redundant=0,
        n_clusters_per_class=1, class_sep=3.0, random_state=0,
    )


def test_all_four_models_present():
    assert set(make_learners()) == MODEL_NAMES


def test_each_model_learns_an_easy_problem():
    X, y = make_easy_data()
    for name, learner in make_learners().items():
        model = learner(X, y)
        assert (model.predict(X) == y).mean() > 0.9, name


def test_same_seed_gives_same_predictions():
    X, y = make_easy_data()
    for name in ["Random Forest", "MLP"]:
        first = make_learners(42)[name](X, y).predict(X)
        second = make_learners(42)[name](X, y).predict(X)
        assert (first == second).all(), name
