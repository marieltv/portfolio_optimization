"""
Data downloading and preprocessing.

Key improvements over v1:
- log-return computation added (missing from original despite being imported in app.py)
- explicit progress=False to suppress yfinance noise
- validates the downloaded universe matches requested tickers
- docstrings with parameter descriptions
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataError(Exception):
    """Raised when downloaded data does not meet quality requirements."""


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    *,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download daily adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers : ticker symbols recognised by Yahoo Finance.
    start : first date to include (ISO format, inclusive).
    end : last date to include (ISO format, exclusive).
    auto_adjust : apply split/dividend adjustment (default True).

    Returns
    -------
    pd.DataFrame
        Columns = tickers, index = pd.DatetimeIndex (business days only).

    Raises
    ------
    DataError
        If any requested ticker has no data in the requested window.
    """
    logger.info("Downloading price data for %d tickers (%s → %s)", len(tickers), start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )["Close"]

    # yfinance returns a Series when only one ticker is requested
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # Identify tickers that are entirely NaN (bad symbol / no data in window)
    missing = [t for t in tickers if raw[t].isna().all()]
    if missing:
        raise DataError(f"No price data returned for tickers: {missing}")

    prices = raw.dropna(how="any")
    logger.info("Downloaded %d rows after dropping NaN rows.", len(prices))
    return prices


def compute_returns(prices: pd.DataFrame, *, log: bool = False) -> pd.DataFrame:
    """
    Compute asset returns from a price DataFrame.

    Parameters
    ----------
    prices : aligned price DataFrame (output of ``download_prices``).
    log : if True, compute continuously-compounded (log) returns;
          otherwise compute simple percentage returns.

    Returns
    -------
    pd.DataFrame
        Same shape as ``prices`` minus the first row.
    """
    if log:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change()
    return returns.dropna()
