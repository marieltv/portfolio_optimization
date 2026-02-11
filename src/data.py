"""
Data downloading and preprocessing utilities.
"""

from typing import List
import pandas as pd
import numpy as np
import yfinance as yf


def download_prices(
    tickers: List[str],
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.
    """
    prices = yf.download(tickers, start=start, end=end)["Close"]
    return prices.dropna()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns.
    """
    return np.log(prices / prices.shift(1)).dropna()
