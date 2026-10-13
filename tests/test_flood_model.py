from sklearn.datasets import make_classification

from flood_model import compare_models, get_models


def test_all_four_models_present():
    assert set(get_models()) == {"Random Forest", "Logistic Regression", "SVC", "MLP"}


def test_compare_models_returns_valid_f1_scores():
    X, y = make_classification(n_samples=200, n_features=3, n_informative=3,
                               n_redundant=0, random_state=0)
    scores = compare_models(X, y)
    for f1 in scores.values():
        assert len(f1) == 2
        assert all(0 <= s <= 1 for s in f1)
