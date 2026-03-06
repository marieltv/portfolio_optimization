"""
Configuration for portfolio optimization.

All parameters live in a frozen dataclass.
This makes configs testable, overridable, and explicitly passed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class BacktestConfig:
    """
    Immutable configuration for a backtest run.

    Parameters
    ----------
    tickers : list of Yahoo Finance ticker symbols.
    start_date / end_date : ISO-format date strings for price history.
    trading_days : trading days per year used for annualisation (252).
    risk_free_rate : annualised risk-free rate for Sharpe/Sortino.
    train_window_years : rolling look-back window in years.
    rebalance_frequency : pandas offset alias ("ME"=month-end, "QE"=quarter).
    max_weight : per-asset upper bound (long-only constraint).
    l2_penalty : L2 regularisation strength for regularised Max-Sharpe.
    """
    tickers: List[str] = field(
        default_factory=lambda: ["BA", "NOC", "LMT", "RTX", "AXON", "GD"]
    )
    start_date: str = "2018-01-01"
    end_date: str = "2026-01-01"
    trading_days: int = 252
    risk_free_rate: float = 0.0
    train_window_years: int = 4
    rebalance_frequency: str = "ME"   # pandas ≥ 2.2: "ME" not deprecated "M"
    max_weight: float = 0.30
    l2_penalty: float = 0.1

    @property
    def train_window_days(self) -> int:
        """Rolling look-back expressed in trading days."""
        return self.train_window_years * self.trading_days

    @property
    def min_weight(self) -> float:
        """Minimum feasible per-asset weight given the universe size."""
        return round(1 / len(self.tickers), 2)
DEFAULT_CONFIG = BacktestConfig()
