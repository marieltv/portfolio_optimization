"""
Performance and risk metrics.
"""

import numpy as np
import pandas as pd
from srs.config import TRADING_DAYS


def annualized_return(returns: pd.Series) -> float:
    return returns.mean() * TRADING_DAYS


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    vol = annualized_volatility(returns)
    if vol == 0:
        return np.nan
    return (annualized_return(returns) - rf) / vol


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def turnover(prev_w: np.ndarray, new_w: np.ndarray) -> float:
    """
    Portfolio turnover between two rebalancing steps.
    """
    return np.sum(np.abs(new_w - prev_w))
