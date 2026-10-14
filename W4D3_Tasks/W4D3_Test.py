
"""Tests for W4D3 model serialisation."""

import os
import pickle
import tempfile
import joblib
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def setup_model():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])
    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_joblib_serialisation():
    model, X_test, _ = setup_model()
    expected = model.predict(X_test)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.joblib")
        joblib.dump(model, path)
        loaded_model = joblib.load(path)

        assert np.array_equal(expected, loaded_model.predict(X_test))


def test_pickle_serialisation():
    model, X_test, _ = setup_model()
    expected = model.predict(X_test)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.pkl")

        with open(path, "wb") as file:
            pickle.dump(model, file)

        with open(path, "rb") as file:
            loaded_model = pickle.load(file)

        assert np.array_equal(expected, loaded_model.predict(X_test))


def test_model_predictions_and_probabilities():
    model, X_test, y_test = setup_model()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert len(predictions) == len(y_test)
    assert probabilities.shape == (len(y_test), 2)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_pipeline_contains_scaler_and_classifier():
    model, _, _ = setup_model()

    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
