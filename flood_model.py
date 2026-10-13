from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_models(seed=42):
    return {
        "Random Forest": RandomForestClassifier(random_state=seed),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVC": SVC(),
        "MLP": MLPClassifier(max_iter=500, random_state=seed),
    }


def compare_models(X, y, seed=42):
    """Train each model on an 80/20 stratified split and return F1 scores."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    scores = {}
    for name, model in get_models(seed).items():
        model.fit(X_train, y_train)
        scores[name] = f1_score(y_test, model.predict(X_test), average=None)
    return scores
