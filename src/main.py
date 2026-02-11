"""
Main execution script.
"""

import pandas as pd
import matplotlib.pyplot as plt

from srs.config import TICKERS, START_DATE, END_DATE
from data import download_prices, compute_log_returns
from backtest import rolling_backtest
from metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
)


def main() -> None:
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    returns = compute_log_returns(prices)

    results = rolling_backtest(returns)

    summary = []
    for name, r in results.items():
        summary.append({
            "Strategy": name,
            "CAGR": annualized_return(r),
            "Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r),
            "MaxDrawdown": max_drawdown(r),
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.round(3))

    plt.figure(figsize=(12, 6))
    for name, r in results.items():
        plt.plot((1 + r).cumprod(), label=name)
    plt.legend()
    plt.title("Out-of-Sample Portfolio Performance")
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
