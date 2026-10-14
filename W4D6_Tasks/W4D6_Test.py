
"""Unit tests for W4D6 Responsible AI utilities."""

import numpy as np
import pandas as pd

from types import SimpleNamespace
from aif360.datasets import BinaryLabelDataset
from W4D6_Responsible_AI import get_model_features, fairness_metrics


def test_protected_feature_is_removed_from_model_inputs():
    dataset = SimpleNamespace(
        feature_names=["age", "sex", "education"],
        features=np.array([
            [25.0, 0.0, 12.0],
            [40.0, 1.0, 16.0],
        ]),
    )

    features, names = get_model_features(dataset)

    assert names == ["age", "education"]
    assert features.shape == (2, 2)
    assert np.array_equal(features[:, 0], [25.0, 40.0])


def test_fairness_metrics_return_expected_keys():
    df = pd.DataFrame({
        "label": [1, 0, 1, 0, 1, 0, 1, 0],
        "sex":   [0, 0, 0, 0, 1, 1, 1, 1],
        "feature": [1, 2, 3, 4, 5, 6, 7, 8],
    })

    dataset = BinaryLabelDataset(
        df=df,
        label_names=["label"],
        protected_attribute_names=["sex"],
        favorable_label=1,
        unfavorable_label=0,
    )

    predictions = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    result = fairness_metrics(dataset, predictions)

    expected_keys = {
        "Disparate Impact",
        "Statistical Parity Difference",
        "Equal Opportunity Difference",
        "Average Odds Difference",
    }

    assert set(result.keys()) == expected_keys


def test_fairness_metrics_are_numeric():
    df = pd.DataFrame({
        "label": [1, 0, 1, 0, 1, 0, 1, 0],
        "sex":   [0, 0, 0, 0, 1, 1, 1, 1],
        "feature": [1, 2, 3, 4, 5, 6, 7, 8],
    })

    dataset = BinaryLabelDataset(
        df=df,
        label_names=["label"],
        protected_attribute_names=["sex"],
        favorable_label=1,
        unfavorable_label=0,
    )

    result = fairness_metrics(dataset, [1, 0, 1, 0, 1, 0, 1, 0])

    for value in result.values():
        assert isinstance(value, (int, float, np.number))


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
