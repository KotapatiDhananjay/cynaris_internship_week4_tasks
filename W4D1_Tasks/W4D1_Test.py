
"""Tests for W4D1 model evaluation."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def setup_model():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def test_predictions_and_probabilities():
    model, _, X_test, _, y_test = setup_model()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_classification_metrics():
    model, _, X_test, _, y_test = setup_model()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    assert 0 <= precision_score(y_test, predictions) <= 1
    assert 0 <= recall_score(y_test, predictions) <= 1
    assert 0 <= f1_score(y_test, predictions) <= 1
    assert 0 <= roc_auc_score(y_test, probabilities) <= 1


def test_stratified_cross_validation():
    data = load_breast_cancer()
    X, y = data.data, data.target

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    assert len(scores) == 5
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0) & (scores <= 1))


def test_scaling_is_inside_pipeline():
    model, *_ = setup_model()

    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
