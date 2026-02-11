"""
Global configuration for portfolio optimization project.
"""

from typing import List

# Assets
TICKERS: List[str] = ["BA", "NOC", "LMT", "RTX", "AXON", "GD"]

# Data range
START_DATE: str = "2018-01-01"
END_DATE: str = "2025-01-01"

# Market assumptions
TRADING_DAYS: int = 252
RISK_FREE_RATE: float = 0.0

# Backtest configuration
TRAIN_WINDOW_YEARS: int = 3
REBALANCE_FREQUENCY: str = "M"  # Monthly
MAX_WEIGHT: float = 0.30

# Regularization
L2_PENALTY: float = 0.1
