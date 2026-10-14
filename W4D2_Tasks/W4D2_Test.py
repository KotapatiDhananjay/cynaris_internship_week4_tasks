
"""Automated tests for W4D2 bias-variance and regularisation."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def setup_data():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_ridge_predictions():
    X_train, X_test, y_train, y_test = setup_data()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert np.all(np.isfinite(predictions))


def test_lasso_predictions():
    X_train, X_test, y_train, y_test = setup_data()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=1.0, max_iter=10000)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert np.all(np.isfinite(predictions))


def test_metrics_are_valid():
    X_train, X_test, y_train, y_test = setup_data()
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    assert np.isfinite(rmse)
    assert rmse >= 0


def test_regularisation_parameters():
    assert Ridge(alpha=1.0).alpha > 0
    assert Lasso(alpha=1.0).alpha > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
