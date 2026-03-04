"""
Streamlit dashboard for the portfolio optimisation backtest.

Run with:
    pip install -e .(locally if you use virtual environment)
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from portfolio_optimization.config import BacktestConfig, DEFAULT_CONFIG
from portfolio_optimization.data import download_prices, compute_returns
from portfolio_optimization.backtest import rolling_backtest, BacktestResult
from portfolio_optimization.metrics import performance_summary

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Optimisation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=", ".join(DEFAULT_CONFIG.tickers),
    help="Yahoo Finance symbols, e.g. AAPL, MSFT, GOOG",
)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start date", value=pd.Timestamp(DEFAULT_CONFIG.start_date))
end_date   = col2.date_input("End date",   value=pd.Timestamp(DEFAULT_CONFIG.end_date))

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest parameters")

train_years = st.sidebar.slider(
    "Training window (years)", min_value=1, max_value=10,
    value=DEFAULT_CONFIG.train_window_years,
)
max_weight = st.sidebar.slider(
    "Max weight per asset", min_value=0.10, max_value=1.0,
    value=DEFAULT_CONFIG.max_weight, step=0.05,
    format="%.0f%%",
)
rebal_freq = st.sidebar.selectbox(
    "Rebalancing frequency",
    options=["ME", "QE", "YE"],
    format_func={"ME": "Monthly", "QE": "Quarterly", "YE": "Yearly"}.get,
    index=0,
)
rf_rate = st.sidebar.number_input(
    "Risk-free rate (annual %)", min_value=0.0, max_value=20.0,
    value=DEFAULT_CONFIG.risk_free_rate * 100, step=0.25,
) / 100

run_button = st.sidebar.button("▶ Run Backtest", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Portfolio Optimisation Dashboard")
st.caption(
    "Compares Equal Weight · Minimum Variance · Risk Parity · Regularised Max Sharpe "
    "on a rolling out-of-sample backtest."
)

# ---------------------------------------------------------------------------
# Colour palette — one consistent colour per strategy across all charts
# ---------------------------------------------------------------------------
STRATEGY_COLORS = {
    "EqualWeight": "#636EFA",
    "MinVariance":  "#EF553B",
    "RiskParity":   "#00CC96",
    "RegMaxSharpe": "#FFA15A",
}

# ---------------------------------------------------------------------------
# Helper: drawdown series
# ---------------------------------------------------------------------------
def _drawdown_series(returns: pd.Series) -> pd.Series:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    return (cumulative - peak) / peak


# ---------------------------------------------------------------------------
# Plotting helpers — all return Plotly figures
# ---------------------------------------------------------------------------

def plot_equity_curves(results: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        equity = (1 + r).cumprod()
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            name=name,
            mode="lines",
            line=dict(color=STRATEGY_COLORS.get(name), width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title="Cumulative Return (out-of-sample)",
        xaxis_title=None,
        yaxis_title="Portfolio value (start = 1)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_drawdowns(results: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        dd = _drawdown_series(r)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            name=name,
            mode="lines",
            fill="tozeroy",
            line=dict(color=STRATEGY_COLORS.get(name), width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title=None,
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".1f",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_rolling_sharpe(
    results: dict[str, pd.Series],
    window: int = 63,
    rf: float = 0.0,
    trading_days: int = 252,
) -> go.Figure:
    fig = go.Figure()
    daily_rf = rf / trading_days
    for name, r in results.items():
        excess = r - daily_rf
        roll_mean = excess.rolling(window).mean() * trading_days
        roll_std  = r.rolling(window).std() * np.sqrt(trading_days)
        roll_sharpe = roll_mean / roll_std.replace(0, np.nan)
        fig.add_trace(go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.values,
            name=name,
            mode="lines",
            line=dict(color=STRATEGY_COLORS.get(name), width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra>" + name + "</extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_layout(
        title=f"Rolling {window}-Day Sharpe Ratio",
        xaxis_title=None,
        yaxis_title="Sharpe ratio",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_weights_over_time(
    weight_history: dict[str, pd.DataFrame],
    tickers: list[str],
) -> go.Figure:
    n_strategies = len(weight_history)
    fig = make_subplots(
        rows=1, cols=n_strategies,
        subplot_titles=list(weight_history.keys()),
        shared_yaxes=True,
    )
    colors = px.colors.qualitative.Pastel[:len(tickers)]
    for col_idx, (name, wdf) in enumerate(weight_history.items(), start=1):
        for t_idx, ticker in enumerate(tickers):
            fig.add_trace(
                go.Bar(
                    x=wdf.index,
                    y=wdf[ticker].values * 100,
                    name=ticker,
                    marker_color=colors[t_idx],
                    showlegend=(col_idx == 1),
                    hovertemplate="%{x|%Y-%m-%d}<br>" + ticker + ": %{y:.1f}%<extra></extra>",
                ),
                row=1, col=col_idx,
            )
    fig.update_layout(
        barmode="stack",
        title="Portfolio Weights Over Time (%)",
        yaxis_title="Weight (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=80, b=0),
        height=380,
    )
    return fig


def plot_return_distribution(results: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Violin(
            y=r.values * 100,
            name=name,
            box_visible=True,
            meanline_visible=True,
            fillcolor=STRATEGY_COLORS.get(name),
            opacity=0.7,
            line_color="white",
            hovertemplate="<b>" + name + "</b><br>%{y:.2f}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_layout(
        title="Daily Return Distribution (%)",
        yaxis_title="Daily return (%)",
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        hovertemplate="%{y} / %{x}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Asset Return Correlation",
        margin=dict(l=0, r=0, t=50, b=0),
        height=350,
    )
    return fig


def style_summary_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Colour-code metrics: green = good, red = bad."""
    pct_cols = ["CAGR", "Volatility", "MaxDrawdown", "CVaR95"]
    styled = df.copy()

    # Format as percentages where appropriate
    fmt = {
        col: "{:.1f}%" for col in pct_cols if col in styled.columns
    }
    fmt.update({
        col: "{:.2f}" for col in styled.columns if col not in pct_cols
    })

    return (
        styled.style
        .format({k: (lambda v, f=f: f.format(v * 100) if "%" in f else f.format(v))
                 for k, f in fmt.items()})
        .background_gradient(subset=["Sharpe", "Sortino", "Calmar"], cmap="Greens")
        .background_gradient(subset=["MaxDrawdown", "CVaR95"], cmap="Reds_r")
        .background_gradient(subset=["CAGR"], cmap="Blues")
    )

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if not run_button:
    st.info("Configure your universe and parameters in the sidebar, then click **▶ Run Backtest**.")
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(tickers) < 2:
    st.error("Please enter at least 2 tickers.")
    st.stop()

cfg = BacktestConfig(
    tickers=tickers,
    start_date=str(start_date),
    end_date=str(end_date),
    train_window_years=train_years,
    max_weight=max_weight,
    rebalance_frequency=rebal_freq,
    risk_free_rate=rf_rate,
)

# --- Download & compute ---
with st.spinner("Downloading price data…"):
    try:
        prices = download_prices(cfg.tickers, cfg.start_date, cfg.end_date)
    except Exception as e:
        st.error(f"Data download failed: {e}")
        st.stop()

returns = compute_returns(prices, log=False)   # simple returns

# --- Run backtest ---
with st.spinner("Running backtest…"):
    backtest: BacktestResult = rolling_backtest(returns, cfg=cfg)

summary = performance_summary(
    backtest.returns,
    rf=cfg.risk_free_rate,
    trading_days=cfg.trading_days,
)

# ---------------------------------------------------------------------------
# Layout — three tabs
# ---------------------------------------------------------------------------
tab_perf, tab_weights, tab_assets = st.tabs([
    "📊 Performance", "⚖️ Weights", "🔍 Asset Analysis"
])

# ============================= TAB 1: Performance ============================
with tab_perf:

    # KPI cards — top row
    best_sharpe = summary["Sharpe"].idxmax()
    best_cagr   = summary["CAGR"].idxmax()
    low_dd      = summary["MaxDrawdown"].idxmax()   # least negative = best

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Best Sharpe",      best_sharpe,  f"{summary.loc[best_sharpe, 'Sharpe']:.2f}")
    kpi2.metric("Best CAGR",        best_cagr,    f"{summary.loc[best_cagr,   'CAGR']*100:.1f}%")
    kpi3.metric("Lowest Drawdown",  low_dd,       f"{summary.loc[low_dd, 'MaxDrawdown']*100:.1f}%")

    st.markdown("---")

    # Summary table
    st.subheader("Performance Summary")
    try:
        st.dataframe(style_summary_table(summary), use_container_width=True)
    except Exception:
        st.dataframe(summary.round(3), use_container_width=True)

    st.markdown("---")

    # Equity curves + drawdown — side by side
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_equity_curves(backtest.returns), use_container_width=True)
    with c2:
        st.plotly_chart(plot_drawdowns(backtest.returns), use_container_width=True)

    # Rolling Sharpe + return distribution — side by side
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            plot_rolling_sharpe(backtest.returns, rf=cfg.risk_free_rate,
                                trading_days=cfg.trading_days),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(plot_return_distribution(backtest.returns), use_container_width=True)

# ============================= TAB 2: Weights ================================
with tab_weights:
    st.subheader("Portfolio Weights Over Time")
    st.plotly_chart(
        plot_weights_over_time(backtest.weights, cfg.tickers),
        use_container_width=True,
    )

    st.subheader("Latest Weights at Last Rebalance")
    latest_cols = st.columns(len(backtest.weights))
    for col, (name, wdf) in zip(latest_cols, backtest.weights.items()):
        latest = wdf.iloc[-1].sort_values(ascending=False)
        fig = go.Figure(go.Bar(
            x=latest.index, y=latest.values * 100,
            marker_color=[STRATEGY_COLORS.get(name, "#888")] * len(latest),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title=name, yaxis_title="Weight (%)",
            margin=dict(l=0, r=0, t=40, b=0), height=280,
        )
        col.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Turnover per Rebalance")
    avg_to = {name: s.mean() for name, s in backtest.turnover.items()}
    fig_to = go.Figure(go.Bar(
        x=list(avg_to.keys()),
        y=[v * 100 for v in avg_to.values()],
        marker_color=[STRATEGY_COLORS.get(n, "#888") for n in avg_to],
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig_to.update_layout(
        yaxis_title="Average one-way turnover (%)",
        margin=dict(l=0, r=0, t=10, b=0), height=300,
    )
    st.plotly_chart(fig_to, use_container_width=True)

# ============================= TAB 3: Asset Analysis =========================
with tab_assets:
    st.subheader("Asset Correlation Matrix")
    st.plotly_chart(plot_correlation_heatmap(returns), use_container_width=True)

    st.subheader("Individual Asset Performance")
    asset_summary = []
    for ticker in cfg.tickers:
        r = returns[ticker]
        asset_summary.append({
            "Ticker":     ticker,
            "CAGR":       r.mean() * cfg.trading_days,
            "Volatility": r.std()  * np.sqrt(cfg.trading_days),
            "Sharpe":     (r.mean() * cfg.trading_days) / (r.std() * np.sqrt(cfg.trading_days)),
            "MaxDrawdown": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(),
        })
    asset_df = pd.DataFrame(asset_summary).set_index("Ticker")
    st.dataframe(asset_df.round(3), use_container_width=True)

    st.subheader("Normalised Price History")
    norm_prices = prices / prices.iloc[0]
    fig_prices = go.Figure()
    for ticker in cfg.tickers:
        fig_prices.add_trace(go.Scatter(
            x=norm_prices.index, y=norm_prices[ticker],
            name=ticker, mode="lines",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>" + ticker + "</extra>",
        ))
    fig_prices.update_layout(
        yaxis_title="Normalised price (start = 1)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_prices, use_container_width=True)
