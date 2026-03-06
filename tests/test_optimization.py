import numpy as np
import pytest

from portfolio_optimization.optimization import (
    EqualWeight,
    MinimumVariance,
    RiskParity,
    RegularisedMaxSharpe,
    OptimizationError,
)

# Shared fixtures
MEAN = np.array([0.10, 0.12, 0.08])
COV = np.array([
    [0.04, 0.008, 0.004],
    [0.008, 0.032, 0.004],
    [0.004, 0.004, 0.028],
])
COV_IDENTITY_4 = np.eye(4)
MEAN_4 = np.array([0.10, 0.12, 0.08, 0.11])


def test_equal_weight_properties():
    w = EqualWeight()(MEAN, COV)
    assert len(w) == 3
    assert np.allclose(w.sum(), 1.0)
    assert np.all(w >= 0)
    assert np.allclose(w, np.ones(3) / 3)


def test_minimum_variance_sums_to_one():
    w = MinimumVariance()(MEAN, COV)
    assert np.allclose(w.sum(), 1.0, atol=1e-6)


def test_minimum_variance_non_negative():
    w = MinimumVariance()(MEAN, COV)
    assert np.all(w >= -1e-8)


def test_minimum_variance_respects_max_weight():
    max_w = 0.5
    w = MinimumVariance(max_weight=max_w)(MEAN, COV)
    assert np.all(w <= max_w + 1e-6)


def test_risk_parity_sums_to_one():
    w = RiskParity()(MEAN_4, COV_IDENTITY_4)
    assert np.allclose(w.sum(), 1.0, atol=1e-6)


def test_risk_parity_identity_equals_equal_weight():
    # With identity covariance all assets have equal variance →
    # risk parity should return equal weights
    w = RiskParity()(MEAN_4, COV_IDENTITY_4)
    assert np.allclose(w, np.ones(4) / 4, atol=1e-2)


def test_regularised_max_sharpe_sums_to_one():
    w = RegularisedMaxSharpe()(MEAN, COV)
    assert np.allclose(w.sum(), 1.0, atol=1e-6)


def test_regularised_max_sharpe_non_negative():
    w = RegularisedMaxSharpe()(MEAN, COV)
    assert np.all(w >= -1e-8)


def test_regularised_max_sharpe_respects_max_weight():
    max_w = 0.5
    w = RegularisedMaxSharpe(max_weight=max_w)(MEAN, COV)
    assert np.all(w <= max_w + 1e-6)


def test_optimization_error_on_bad_cov():
    bad_cov = np.array([[1, 2], [3, 4]])   # not symmetric
    mean = np.array([0.1, 0.1])
    with pytest.raises((OptimizationError, ValueError)):
        MinimumVariance()(mean, bad_cov)