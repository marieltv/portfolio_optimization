"""
CLI entry point for a headless backtest run.

Usage:
    python -m portfolio_optimization.main
    python -m portfolio_optimization.main --tickers AAPL MSFT GOOG --start 2015-01-01
"""
from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd

from portfolio_optimization.config import BacktestConfig, DEFAULT_CONFIG
from portfolio_optimization.data import download_prices, compute_returns
from portfolio_optimization.backtest import rolling_backtest
from portfolio_optimization.metrics import performance_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portfolio optimisation backtest")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_CONFIG.tickers)
    p.add_argument("--start",   default=DEFAULT_CONFIG.start_date)
    p.add_argument("--end",     default=DEFAULT_CONFIG.end_date)
    p.add_argument("--no-plot", action="store_true", help="Skip equity-curve plot")
    return p.parse_args()


def run(cfg: BacktestConfig, plot: bool = True) -> pd.DataFrame:
    """
    Execute a full backtest and return the performance summary DataFrame.

    Parameters
    ----------
    cfg : fully specified backtest configuration.
    plot : whether to display the equity-curve chart.

    Returns
    -------
    pd.DataFrame : performance metrics (CAGR, Sharpe, MDD, …) per strategy.
    """
    prices  = download_prices(cfg.tickers, cfg.start_date, cfg.end_date)
    returns = compute_returns(prices)

    backtest = rolling_backtest(returns, cfg=cfg)
    summary  = performance_summary(backtest.returns, rf=cfg.risk_free_rate,
                                   trading_days=cfg.trading_days)

    print("\n=== Performance Summary ===")
    print(summary.round(3).to_string())

    if plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, r in backtest.returns.items():
            (1 + r).cumprod().plot(ax=ax, label=name)
        ax.set_title("Out-of-Sample Portfolio Performance")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return summary


def main() -> None:
    args = parse_args()
    cfg = BacktestConfig(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
    )
    run(cfg, plot=not args.no_plot)


if __name__ == "__main__":
    main()
