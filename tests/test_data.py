import numpy as np
import pandas as pd

from src.data import compute_returns, clean_data

def test_compute_returns_correctness():
    prices = pd.DataFrame({
        "A": [100, 110, 121],
        "B": [50, 55, 60.5],
    })

    returns = compute_returns(prices)

    expected = pd.DataFrame({
        "A": [0.10, 0.10],
        "B": [0.10, 0.10],
    })

    assert returns.shape == expected.shape
    assert np.allclose(returns.values, expected.values)

def test_clean_data_removes_nans():
    df = pd.DataFrame({
        "A": [1.0, np.nan, 3.0],
        "B": [1.0, 2.0, np.nan],
    })

    cleaned = clean_data(df)

    assert not cleaned.isna().any().any()

def test_compute_returns_shape():
    prices = pd.DataFrame(
        np.random.rand(100, 4),
        columns=["A", "B", "C", "D"]
    )

    returns = compute_returns(prices)

    assert returns.shape[0] == prices.shape[0] - 1
    assert returns.shape[1] == prices.shape[1]

def test_returns_index_alignment():
    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame(
        np.arange(10).reshape(5, 2) + 100,
        index=dates,
        columns=["A", "B"],
    )

    returns = compute_returns(prices)

    assert returns.index.equals(dates[1:])
def test_returns_zero_when_prices_constant():
    prices = pd.DataFrame({
        "A": [100, 100, 100],
        "B": [50, 50, 50],
    })

    returns = compute_returns(prices)

    assert np.allclose(returns.values, 0.0)

