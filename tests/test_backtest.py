import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.backtest import rolling_backtest, BacktestResult
from portfolio_optimization.config import BacktestConfig


def _make_returns(n_days: int = 800, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily returns with a realistic date index."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)  # business days only
    data = rng.normal(0.0005, 0.01, size=(n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


# Minimal config: 1-year training window so tests run fast
FAST_CFG = BacktestConfig(
    tickers=["A0", "A1", "A2"],
    train_window_years=1,
    rebalance_frequency="QE",   # quarterly → fewer rebalances → faster
)


def test_returns_result_type():
    returns = _make_returns()
    result = rolling_backtest(returns, cfg=FAST_CFG)
    assert isinstance(result, BacktestResult)
    for name, s in result.returns.items():
        assert isinstance(s, pd.Series), f"{name} returns should be a Series"


def test_oos_starts_after_training_window():
    returns = _make_returns()
    result = rolling_backtest(returns, cfg=FAST_CFG)
    first_oos = min(s.index[0] for s in result.returns.values())
    first_valid = returns.index[FAST_CFG.train_window_days]
    assert first_oos >= first_valid, "OOS period must not start before training window ends"


def test_no_nan_in_returns():
    returns = _make_returns()
    result = rolling_backtest(returns, cfg=FAST_CFG)
    for name, s in result.returns.items():
        assert not s.isna().any(), f"{name} has NaN in returns"


def test_weights_sum_to_one():
    returns = _make_returns()
    result = rolling_backtest(returns, cfg=FAST_CFG)
    for name, wdf in result.weights.items():
        row_sums = wdf.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"{name} weights don't sum to 1"


def test_weights_non_negative():
    returns = _make_returns()
    result = rolling_backtest(returns, cfg=FAST_CFG)
    for name, wdf in result.weights.items():
        assert (wdf.values >= -1e-8).all(), f"{name} has negative weights"


def test_all_strategies_present():
    returns = _make_returns()
    result = rolling_backtest(returns, cfg=FAST_CFG)
    expected = {"EqualWeight", "MinVariance", "RiskParity", "RegMaxSharpe"}
    assert set(result.returns.keys()) == expected


def test_predicted_keys_match_returns():
    returns = _make_returns()
    result = rolling_backtest(returns, cfg=FAST_CFG)
    assert set(result.predicted.keys()) == set(result.returns.keys())


def test_predicted_columns_present():
    returns = _make_returns()
    result = rolling_backtest(returns, cfg=FAST_CFG)
    required_cols = {"pred_sharpe", "pred_sortino", "pred_vol",
                     "actual_sharpe", "actual_sortino", "actual_vol"}
    for name, df in result.predicted.items():
        assert required_cols.issubset(df.columns), f"{name} missing predicted columns"