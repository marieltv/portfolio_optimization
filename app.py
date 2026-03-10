"""
Streamlit dashboard for the portfolio optimisation backtest.

Run with:
    pip install -e .(locally if you use virtual environment)
    streamlit run app.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from portfolio_optimization.backtest import BacktestResult, rolling_backtest
from portfolio_optimization.config import DEFAULT_CONFIG, BacktestConfig
from portfolio_optimization.data import download_prices, compute_returns
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
# Colour palette — one consistent colour per strategy across all charts
# ---------------------------------------------------------------------------
STRATEGY_COLORS = {
    "EqualWeight": "#636EFA",
    "MinVariance":  "#EF553B",
    "RiskParity":   "#00CC96",
    "RegMaxSharpe": "#FFA15A",
}

# ---------------------------------------------------------------------------
# Sidebar
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
    "Training window (years)", min_value=1, max_value=7,
    value=DEFAULT_CONFIG.train_window_years,
)

# parse tickers first, before the slider
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# temporary config just to get min_weight
n = max(len(tickers), 1)
min_weight = round(1 / n, 2)

max_weight_pct = st.sidebar.slider(
    "Max weight per asset (%)",
    min_value=int(min_weight * 100),
    max_value=50,
    value=max(int(DEFAULT_CONFIG.max_weight * 100), int(min_weight * 100)),
    step=5,
    format="%d%%",
)
st.sidebar.caption(f"Min {int(min_weight * 100)}% with {n} assets")
max_weight = max_weight_pct / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Shrinkage estimators")

use_ledoit_wolf = st.sidebar.toggle(
    "Ledoit-Wolf (covariance)",
    value=False,
    help="Shrinks sample covariance toward a structured target. "
         "Reduces estimation error, improves volatility prediction.",
)
use_james_stein = st.sidebar.toggle(
    "James-Stein (mean returns)",
    value=False,
    help="Shrinks individual asset means toward the grand mean. "
         "Reduces impact of extreme return estimates on RegMaxSharpe.",
)

rebal_freq = st.sidebar.selectbox(
    "Rebalancing frequency",
    options=["ME", "QE", "YE"],
    format_func={"ME": "Monthly", "QE": "Quarterly", "YE": "Yearly"}.get,
    index=0,
)

st.sidebar.markdown("---")
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
# Helpers
# ---------------------------------------------------------------------------

def _drawdown_series(returns: pd.Series) -> pd.Series:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    return (cumulative - peak) / peak


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def plot_equity_curves(results: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        equity = (1 + r).cumprod()
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values, name=name, mode="lines",
            line=dict(color=STRATEGY_COLORS.get(name), width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title="Cumulative Return (out-of-sample)", yaxis_title="Portfolio value (start = 1)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=60),
    )
    return fig


def plot_drawdowns(results: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        dd = _drawdown_series(r)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100, name=name, mode="lines", fill="tozeroy",
            line=dict(color=STRATEGY_COLORS.get(name), width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title="Drawdown (%)", yaxis_title="Drawdown (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=60),
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
        roll_sharpe = (
            excess.rolling(window).mean() * trading_days
            / (r.rolling(window).std() * np.sqrt(trading_days)).replace(0, np.nan)
        )
        fig.add_trace(go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.values, name=name, mode="lines",
            line=dict(color=STRATEGY_COLORS.get(name), width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra>" + name + "</extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_layout(
        title=f"Rolling {window}-Day Sharpe Ratio", yaxis_title="Sharpe ratio",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=60),
    )
    return fig


def plot_return_distribution(results: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Violin(
            y=r.values * 100, name=name, box_visible=True, meanline_visible=True,
            fillcolor=STRATEGY_COLORS.get(name), opacity=0.7, line_color="white",
            hovertemplate="<b>" + name + "</b><br>%{y:.2f}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_layout(
        title="Daily Return Distribution (%)", yaxis_title="Daily return (%)",
        showlegend=False, margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_metric_comparison(
    summary: pd.DataFrame,
    metrics: list[str],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for strategy in summary.index:
        fig.add_trace(go.Bar(
            name=strategy,
            x=metrics,
            y=summary.loc[strategy, metrics].values,
            marker_color=STRATEGY_COLORS.get(strategy),
            hovertemplate="%{x}: %{y:.3f}<extra>" + strategy + "</extra>",
        ))
    fig.update_layout(
        barmode="group", title=title,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=60),
    )
    return fig


def plot_weights_over_time(
    weight_history: dict[str, pd.DataFrame],
    tickers: list[str],
) -> go.Figure:
    n = len(weight_history)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=list(weight_history.keys()),
        shared_yaxes=True,
    )
    colors = px.colors.qualitative.Pastel[:len(tickers)]
    for col_idx, (name, wdf) in enumerate(weight_history.items(), start=1):
        for t_idx, ticker in enumerate(tickers):
            fig.add_trace(
                go.Bar(
                    x=wdf.index, y=wdf[ticker].values * 100,
                    name=ticker, marker_color=colors[t_idx],
                    showlegend=(col_idx == 1),
                    hovertemplate="%{x|%Y-%m-%d}<br>" + ticker + ": %{y:.1f}%<extra></extra>",
                ),
                row=1, col=col_idx,
            )
    fig.update_layout(
        barmode="stack", yaxis_title="Weight (%)",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=60), height=380,
    )
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}",
        hovertemplate="%{y} / %{x}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Asset Return Correlation",
        margin=dict(l=0, r=0, t=50, b=0), height=350,
    )
    return fig


def plot_predicted_vs_actual(
    predicted: dict[str, pd.DataFrame],
    metric: str,
    title: str,
    y_label: str,
) -> go.Figure:
    """
    For each strategy: dashed line = predicted, solid line = actual.
    Shaded area fills the gap between them.

    metric must be one of: "sharpe", "sortino", "vol"
    """
    pred_col   = f"pred_{metric}"
    actual_col = f"actual_{metric}"

    fig = go.Figure()

    for name, df in predicted.items():
        color = STRATEGY_COLORS.get(name, "#888")

        # Predicted — dashed
        fig.add_trace(go.Scatter(
            x=df.index, y=df[pred_col],
            name=f"{name} predicted",
            mode="lines",
            line=dict(color=color, dash="dash", width=1.5),
            legendgroup=name,
            hovertemplate="%{x|%Y-%m-%d}<br>Predicted: %{y:.2f}<extra>" + name + "</extra>",
        ))

        # Actual — solid
        fig.add_trace(go.Scatter(
            x=df.index, y=df[actual_col],
            name=f"{name} actual",
            mode="lines",
            line=dict(color=color, width=2.5),
            legendgroup=name,
            hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:.2f}<extra>" + name + "</extra>",
        ))

        # Shaded gap
        x_fill = list(df.index) + list(df.index[::-1])
        y_fill = list(df[pred_col].fillna(0)) + list(df[actual_col].fillna(0)[::-1])
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill,
            fill="toself", fillcolor=color, opacity=0.08,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            legendgroup=name,
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    fig.update_layout(
        title=title, yaxis_title=y_label,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=80),
        height=420,
    )
    return fig

def prediction_error_table(predicted: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in predicted.items():
        rows.append({
            "Strategy":           name,
            "Sharpe Spearman":    df["pred_sharpe"].corr(df["actual_sharpe"],   method="spearman"),
            "Sharpe Bias":        (df["pred_sharpe"]  - df["actual_sharpe"]).mean(),
            "Sharpe MAE":         (df["pred_sharpe"]  - df["actual_sharpe"]).abs().mean(),
            "Sortino Spearman":   df["pred_sortino"].corr(df["actual_sortino"], method="spearman"),
            "Sortino Bias":       (df["pred_sortino"] - df["actual_sortino"]).mean(),
            "Sortino MAE":        (df["pred_sortino"] - df["actual_sortino"]).abs().mean(),
            "Vol Spearman":       df["pred_vol"].corr(df["actual_vol"],         method="spearman"),
            "Vol Bias":           (df["pred_vol"]     - df["actual_vol"]).mean(),
            "Vol MAE":            (df["pred_vol"]     - df["actual_vol"]).abs().mean(),
        })
    return pd.DataFrame(rows).set_index("Strategy")

def style_summary_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    pct_cols = ["CAGR", "Volatility", "MaxDrawdown", "CVaR95"]

    def fmt(col):
        if col in pct_cols:
            return lambda v: f"{v * 100:.1f}%"
        return lambda v: f"{v:.2f}"

    formatters = {col: fmt(col) for col in df.columns}
    styler = df.style.format(formatters)
    if "Sharpe" in df.columns and "Sortino" in df.columns and "Calmar" in df.columns:
        styler = styler.background_gradient(subset=["Sharpe", "Sortino", "Calmar"], cmap="Greens")
    if "MaxDrawdown" in df.columns and "CVaR95" in df.columns:
        risk_cols = [c for c in ["Volatility", "MaxDrawdown", "CVaR95"] if c in df.columns]
        styler = styler.background_gradient(subset=risk_cols, cmap="Reds_r")
    if "CAGR" in df.columns:
        styler = styler.background_gradient(subset=["CAGR"], cmap="Blues")
    return styler


# ---------------------------------------------------------------------------
# Gate
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
    use_ledoit_wolf=use_ledoit_wolf,  
    use_james_stein=use_james_stein,
)

# ---------------------------------------------------------------------------
# Download & backtest
# ---------------------------------------------------------------------------
with st.spinner("Downloading price data…"):
    try:
        prices = download_prices(cfg.tickers, cfg.start_date, cfg.end_date)
    except Exception as e:
        st.error(f"Data download failed: {e}")
        st.stop()

returns = compute_returns(prices)

with st.spinner("Running backtest…"):
    backtest: BacktestResult = rolling_backtest(returns, cfg=cfg)

summary = performance_summary(
    backtest.returns,
    rf=cfg.risk_free_rate,
    trading_days=cfg.trading_days,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_perf, tab_pred, tab_weights, tab_assets = st.tabs([
    "📊 Performance",
    "🎯 Predicted vs Actual",
    "⚖️ Weights",
    "🔍 Asset Analysis",
])

# ================================================================ TAB 1: Performance
with tab_perf:

    best_sharpe = summary["Sharpe"].idxmax()
    best_cagr   = summary["CAGR"].idxmax()
    low_dd      = summary["MaxDrawdown"].idxmax()

    k1, k2, k3 = st.columns(3)
    k1.metric("Best Sharpe",     best_sharpe, f"{summary.loc[best_sharpe, 'Sharpe']:.2f}")
    k2.metric("Best CAGR",       best_cagr,   f"{summary.loc[best_cagr,   'CAGR']*100:.1f}%")
    k3.metric("Lowest Drawdown", low_dd,      f"{summary.loc[low_dd, 'MaxDrawdown']*100:.1f}%")

    st.markdown("---")
    st.subheader("Performance Summary")
    try:
        st.dataframe(style_summary_table(summary), use_container_width=True)
    except Exception:
        st.dataframe(summary.round(3), use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_equity_curves(backtest.returns), use_container_width=True)
    with c2:
        st.plotly_chart(plot_drawdowns(backtest.returns), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            plot_rolling_sharpe(backtest.returns, rf=cfg.risk_free_rate,
                                trading_days=cfg.trading_days),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(plot_return_distribution(backtest.returns), use_container_width=True)

    st.markdown("---")
    st.subheader("Strategy Comparison")
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(plot_metric_comparison(
            summary,
            metrics=["CAGR", "Sharpe", "Sortino", "Calmar"],
            title="Return Quality Metrics",
        ), use_container_width=True)
    with c6:
        st.plotly_chart(plot_metric_comparison(
            summary,
            metrics=["Volatility", "MaxDrawdown", "CVaR95"],
            title="Risk Metrics",
        ), use_container_width=True)

# ================================================================ TAB 2: Predicted vs Actual
with tab_pred:
    st.subheader("🎯 Predicted vs Actual — per Rebalance Period")
    # Summary table first — the answer at a glance
    st.subheader("Estimation Error Summary")
    err_df = prediction_error_table(backtest.predicted)
    st.dataframe(
        err_df.style
            .background_gradient(
                subset=["Sharpe Spearman", "Sortino Spearman", "Vol Spearman"],
                cmap="Greens"
            )
            .background_gradient(
                subset=["Sharpe MAE", "Sortino MAE", "Vol MAE"],
                cmap="Reds"
            )
            .background_gradient(
                subset=["Sharpe Bias", "Sortino Bias", "Vol Bias"],
                cmap="RdYlGn_r"
            )
            .format("{:.3f}"),
        use_container_width=True,
    )
    st.caption("MAE — average absolute prediction error. Bias — positive means optimizer was systematically optimistic.")

    st.caption(
        "**Dashed** = what the optimiser expected going into each period.  "
        "**Solid** = what actually happened.  "
        "The shaded gap reveals how well the optimiser's assumptions held up out-of-sample."
    )

    st.markdown("---")

    # Sharpe — full width, most important metric
    st.plotly_chart(
        plot_predicted_vs_actual(
            backtest.predicted,
            metric="sharpe",
            title="Sharpe Ratio — Predicted vs Actual",
            y_label="Sharpe ratio",
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # Sortino + Volatility side by side
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_predicted_vs_actual(
                backtest.predicted,
                metric="sortino",
                title="Sortino Ratio — Predicted vs Actual",
                y_label="Sortino ratio",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            plot_predicted_vs_actual(
                backtest.predicted,
                metric="vol",
                title="Volatility — Predicted vs Actual",
                y_label="Annualised volatility",
            ),
            use_container_width=True,
        )

# ================================================================ TAB 3: Weights
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
            marker_color=STRATEGY_COLORS.get(name, "#888"),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            yaxis_title="Weight (%)",
            margin=dict(l=0, r=0, t=10, b=0), height=280,  # t=10 since no title anymore
        )
        col.markdown(f"**{name}**")   # strategy name above the chart
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

# ================================================================ TAB 4: Asset Analysis
with tab_assets:
    st.subheader("Asset Correlation Matrix")
    st.plotly_chart(plot_correlation_heatmap(returns), use_container_width=True)

    st.subheader("Individual Asset Performance")
    asset_rows = []
    for ticker in cfg.tickers:
        r = returns[ticker]
        asset_rows.append({
            "Ticker":      ticker,
            "CAGR":        r.mean() * cfg.trading_days,
            "Volatility":  r.std()  * np.sqrt(cfg.trading_days),
            "Sharpe":      (r.mean() * cfg.trading_days) / (r.std() * np.sqrt(cfg.trading_days)),
            "MaxDrawdown": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(),
        })
    asset_df = pd.DataFrame(asset_rows).set_index("Ticker")
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
