import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.data import compute_returns


def test_compute_returns_correctness():
    prices = pd.DataFrame({
        "A": [100.0, 110.0, 121.0],
        "B": [50.0,  55.0,  60.5],
    })
    returns = compute_returns(prices)
    expected = pd.DataFrame({"A": [0.10, 0.10], "B": [0.10, 0.10]})
    assert returns.shape == expected.shape
    assert np.allclose(returns.values, expected.values)


def test_compute_returns_shape():
    prices = pd.DataFrame(np.random.rand(100, 4), columns=["A", "B", "C", "D"])
    returns = compute_returns(prices)
    assert returns.shape == (99, 4)


def test_compute_returns_index_alignment():
    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame(
        np.arange(10).reshape(5, 2) + 100.0,
        index=dates,
        columns=["A", "B"],
    )
    returns = compute_returns(prices)
    assert returns.index.equals(dates[1:])


def test_compute_returns_zero_when_prices_constant():
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0], "B": [50.0, 50.0, 50.0]})
    returns = compute_returns(prices)
    assert np.allclose(returns.values, 0.0)


def test_compute_returns_no_nan():
    prices = pd.DataFrame(np.random.rand(50, 3) + 1, columns=["A", "B", "C"])
    returns = compute_returns(prices)
    assert not returns.isna().any().any()


def test_compute_returns_drops_first_row_nan():
    # pct_change produces NaN in row 0 — compute_returns must drop it
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0]})
    returns = compute_returns(prices)
    assert len(returns) == 2
    assert not returns.isna().any().any()

