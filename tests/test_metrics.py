import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    sharpe_ratio,
    annualized_return,
    annualized_volatility,
    max_drawdown,
)


def test_annualized_return():
    returns = pd.Series([0.01, 0.01, 0.01])
    expected = 0.01 * 252
    assert np.isclose(annualized_return(returns), expected)


def test_annualized_volatility():
    returns = pd.Series([0.01, -0.01] * 100)
    expected = returns.std() * np.sqrt(252)
    assert np.isclose(annualized_volatility(returns), expected)


def test_sharpe_ratio_zero_volatility():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert sharpe_ratio(returns, rf=0.0) == 0.0


def test_max_drawdown_simple_case():
    returns = pd.Series([0.1, -0.2, 0.05])
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    expected = drawdown.min()
    assert np.isclose(max_drawdown(returns), expected)
