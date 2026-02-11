import numpy as np
import pandas as pd

from src.backtest import rolling_backtest
from src.optimization import equal_weight


def test_backtest_start_after_window():
    dates = pd.date_range("2020-01-01", periods=300)
    returns = pd.DataFrame(
        np.random.normal(0, 0.01, size=(300, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )

    window = 252

    portfolio_returns = rolling_backtest(
        returns=returns,
        window=window,
        optimizer=lambda mean, cov: equal_weight(3),
    )

    assert len(portfolio_returns) == len(returns) - window


def test_backtest_no_nan_output():
    dates = pd.date_range("2020-01-01", periods=300)
    returns = pd.DataFrame(
        np.random.normal(0, 0.01, size=(300, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )

    portfolio_returns = rolling_backtest(
        returns=returns,
        window=50,
        optimizer=lambda mean, cov: equal_weight(3),
    )

    assert not portfolio_returns.isna().any()
