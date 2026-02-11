import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from portfolio_optimization.config import TICKERS, START_DATE, END_DATE
from portfolio_optimization.data import download_prices, compute_log_returns
from portfolio_optimization.backtest import rolling_backtest
from portfolio_optimization.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
)

st.set_page_config(page_title="Portfolio Optimization", layout="wide")

st.title("Robust Portfolio Optimization Dashboard")

st.sidebar.header("Settings")

tickers = st.sidebar.text_input(
    "Tickers (comma separated)",
    value=",".join(TICKERS)
)

tickers = [t.strip() for t in tickers.split(",")]

if st.sidebar.button("Run Backtest"):

    with st.spinner("Downloading data..."):
        prices = download_prices(tickers, START_DATE, END_DATE)
        returns = compute_log_returns(prices)

    with st.spinner("Running backtest..."):
        results, weights = rolling_backtest(returns)

    st.subheader("Performance Summary")

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
    st.dataframe(summary_df.round(3))

    st.subheader("Equity Curves")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in results.items():
        ax.plot((1 + r).cumprod(), label=name)

    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Latest Portfolio Weights")

    for name, w in weights.items():
        st.write(f"### {name}")
        st.bar_chart(w.iloc[-1])
