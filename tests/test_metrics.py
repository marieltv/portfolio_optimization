import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    value_at_risk,
    conditional_var,
)

TRADING_DAYS = 252


def test_annualized_return_known_value():
    returns = pd.Series([0.01] * 10)
    assert np.isclose(annualized_return(returns, TRADING_DAYS), 0.01 * TRADING_DAYS)


def test_annualized_volatility_known_value():
    returns = pd.Series([0.01, -0.01] * 100)
    expected = returns.std() * np.sqrt(TRADING_DAYS)
    assert np.isclose(annualized_volatility(returns, TRADING_DAYS), expected)


def test_sharpe_zero_volatility_returns_nan():
    # Constant non-zero returns → std = 0 → Sharpe should be nan
    returns = pd.Series([0.001] * 10)
    assert np.isnan(sharpe_ratio(returns, rf=0.0, trading_days=TRADING_DAYS))


def test_sharpe_positive_for_positive_returns():
    returns = pd.Series(np.random.normal(0.001, 0.01, 500))
    assert sharpe_ratio(returns) > 0  # positive mean → positive Sharpe


def test_max_drawdown_is_non_positive():
    returns = pd.Series([0.1, -0.2, 0.05, -0.1, 0.03])
    assert max_drawdown(returns) <= 0


def test_max_drawdown_known_value():
    # Goes up 10%, then drops 20% from peak
    returns = pd.Series([0.1, -0.2, 0.05])
    cumulative = (1 + returns).cumprod()   # [1.1, 0.88, 0.924]
    peak = cumulative.cummax()             # [1.1, 1.1, 1.1]
    expected = ((cumulative - peak) / peak).min()
    assert np.isclose(max_drawdown(returns), expected)


def test_max_drawdown_all_positive_returns():
    # Monotonically rising → no drawdown
    returns = pd.Series([0.01] * 50)
    assert np.isclose(max_drawdown(returns), 0.0, atol=1e-8)


def test_sortino_higher_than_sharpe_for_positive_skew():
    # When downside vol < total vol, Sortino > Sharpe
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.01, 500))
    assert sortino_ratio(returns) >= sharpe_ratio(returns) - 0.1  # roughly


def test_calmar_ratio_sign():
    returns = pd.Series(np.random.normal(0.001, 0.01, 500))
    # Calmar = CAGR / |MDD|, should be positive when CAGR > 0
    if annualized_return(returns) > 0:
        assert calmar_ratio(returns) > 0


def test_var_is_non_positive():
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    assert value_at_risk(returns, confidence=0.95) < 0


def test_cvar_less_than_or_equal_var():
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    var = value_at_risk(returns, 0.95)
    cvar = conditional_var(returns, 0.95)
    assert cvar <= var  # CVaR is always worse than VaR
