"""
portfolio_optimization
~~~~~~~~~~~~~~~~~~~~~~
Robust multi-strategy portfolio optimisation with rolling backtest.
"""
from portfolio_optimization.config import BacktestConfig, DEFAULT_CONFIG
from portfolio_optimization.backtest import rolling_backtest, BacktestResult
from portfolio_optimization.metrics import performance_summary

__all__ = [
    "BacktestConfig",
    "DEFAULT_CONFIG",
    "rolling_backtest",
    "BacktestResult",
    "performance_summary",
]