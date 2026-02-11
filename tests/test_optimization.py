import numpy as np
import pytest

from src.optimization import (
    equal_weight,
    minimum_variance,
    risk_parity,
    regularized_max_sharpe,
)


def test_equal_weight_properties():
    n = 5
    weights = equal_weight(n)

    assert len(weights) == n
    assert np.allclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)
    assert np.allclose(weights, np.ones(n) / n)


def test_minimum_variance_constraints():
    mean = np.array([0.1, 0.12, 0.08])
    cov = np.array([
        [0.1, 0.02, 0.01],
        [0.02, 0.08, 0.01],
        [0.01, 0.01, 0.07],
    ])

    weights = minimum_variance(mean, cov)

    assert np.allclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= 0)


def test_risk_parity_equal_risk_contributions():
    cov = np.eye(4)  # identity → equal variance assets
    weights = risk_parity(cov)

    # In identity case, should approximate equal weights
    assert np.allclose(weights, np.ones(4) / 4, atol=1e-2)


def test_regularized_max_sharpe_constraints():
    mean = np.array([0.1, 0.15, 0.05])
    cov = np.eye(3)

    weights = regularized_max_sharpe(mean, cov, l2_reg=0.1)

    assert np.allclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= 0)
