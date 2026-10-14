
"""Basic automated tests for the W4D5 classification capstone."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_data():
    dataset = load_breast_cancer()
    return train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )


def test_logistic_regression_metrics():
    X_train, X_test, y_train, y_test = get_data()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    assert 0 <= accuracy_score(y_test, predictions) <= 1
    assert 0 <= precision_score(
        y_test, predictions, zero_division=0
    ) <= 1
    assert 0 <= recall_score(
        y_test, predictions, zero_division=0
    ) <= 1
    assert 0 <= roc_auc_score(y_test, probabilities) <= 1


def test_random_forest_predictions():
    X_train, X_test, y_train, y_test = get_data()
    model = RandomForestClassifier(
        n_estimators=50, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert np.all(np.isfinite(probabilities))


def test_probability_values():
    X_train, X_test, y_train, _ = get_data()
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_both_models_can_be_trained():
    X_train, _, y_train, _ = get_data()

    logistic = LogisticRegression(max_iter=2000)
    forest = RandomForestClassifier(
        n_estimators=50, random_state=42
    )

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    assert hasattr(logistic, "classes_")
    assert hasattr(forest, "classes_")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
