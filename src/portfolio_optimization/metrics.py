"""
Performance and risk metrics.

All functions are *pure* — they take a returns Series and return a scalar.
This makes them trivially testable and composable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from portfolio_optimization.config import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def annualized_return(
    returns: pd.Series,
    trading_days: int = DEFAULT_CONFIG.trading_days,
) -> float:
    """Arithmetic annualisation of mean daily return (CAGR approximation)."""
    return float(returns.mean() * trading_days)


def annualized_volatility(returns: pd.Series,  trading_days: int = DEFAULT_CONFIG.trading_days) -> float:
    """Annualised standard deviation of daily returns."""
    return float(returns.std() * np.sqrt(trading_days))


def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    trading_days: int = DEFAULT_CONFIG.trading_days
) -> float:
    """
    Annualised Sharpe ratio.

    Returns ``np.nan`` when volatility is effectively zero.
    """
    vol = annualized_volatility(returns, trading_days)
    if vol < 1e-12:
        return np.nan
    return (annualized_return(returns, trading_days) - rf) / vol


def sortino_ratio(
    returns: pd.Series,
    rf: float = 0.0,
     trading_days: int = DEFAULT_CONFIG.trading_days
) -> float:
    """
    Annualised Sortino ratio (downside deviation denominator).

    Uses semi-deviation: only days below the risk-free daily rate count.
    """
    daily_rf = rf / trading_days
    downside = returns[returns < daily_rf] - daily_rf
    downside_vol = float(np.sqrt((downside ** 2).mean()) * np.sqrt(trading_days))
    if downside_vol < 1e-12:
        return np.nan
    excess = annualized_return(returns, trading_days) - rf
    return excess / downside_vol


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (negative number, e.g. -0.35 = -35%).
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    return float(((cumulative - peak) / peak).min())


def calmar_ratio(returns: pd.Series,  trading_days: int = DEFAULT_CONFIG.trading_days) -> float:
    """CAGR divided by absolute max drawdown."""
    mdd = abs(max_drawdown(returns))
    if mdd < 1e-12:
        return np.nan
    return annualized_return(returns, trading_days) / mdd


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (negative number)."""
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR / Expected Shortfall (average loss beyond VaR)."""
    var = value_at_risk(returns, confidence)
    return float(returns[returns <= var].mean())


def turnover(prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
    """
    One-way portfolio turnover between two rebalancing steps.

    Returns the sum of absolute weight changes; 2.0 = complete turnover.
    """
    return float(np.sum(np.abs(new_weights - prev_weights)))


# ---------------------------------------------------------------------------
# Summary table helper
# ---------------------------------------------------------------------------

def performance_summary(
    results: dict[str, pd.Series],
    rf: float = 0.0,
    trading_days: int = DEFAULT_CONFIG.trading_days,
) -> pd.DataFrame:
    """
    Build a tidy summary DataFrame for a dict of strategy return series.

    Parameters
    ----------
    results : mapping of strategy name → daily return Series.
    rf : annualised risk-free rate.
    trading_days : trading days per year.

    Returns
    -------
    pd.DataFrame with columns:
        CAGR, Volatility, Sharpe, Sortino, MaxDrawdown, Calmar, CVaR95
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "Strategy":    name,
            "CAGR":        annualized_return(r, trading_days),
            "Volatility":  annualized_volatility(r, trading_days),
            "Sharpe":      sharpe_ratio(r, rf, trading_days),
            "Sortino":     sortino_ratio(r, rf, trading_days),
            "MaxDrawdown": max_drawdown(r),
            "Calmar":      calmar_ratio(r, trading_days),
            "CVaR95":      conditional_var(r),
        })
    return pd.DataFrame(rows).set_index("Strategy")