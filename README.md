# Out-of-Sample Evaluation of Portfolio Optimization Methods Under Estimation Uncertainty

[![CI](https://github.com/marieltv/portfolio_optimization/actions/workflows/ci.yaml/badge.svg)](https://github.com/marieltv/portfolio_optimization/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/live%20demo-streamlit-ff4b4b)](https://portfolio-optimisation.streamlit.app)

## 1. Business Problem

Asset managers and quantitative analysts face a consistent challenge: selecting an allocation strategy that is robust out-of-sample, not just historically attractive. Mean-variance optimisation has been the industry standard since the work of Harry Markowitz (1952), yet in practice it is well-documented to be highly sensitive to estimation error — small changes in expected return inputs produce dramatically different, often extreme portfolios.

This project addresses two operational questions directly relevant to portfolio management:

1. **Which allocation strategy delivers the best risk-adjusted return when deployed forward in time** — not fitted to history?
2. **How accurately does each strategy predict its own performance** — and which strategies should be trusted when their assumptions are violated by regime change?

The predicted vs actual framework makes estimation error visible and measurable, rather than burying it in aggregate backtest statistics. This is directly applicable to strategy selection, risk budgeting, and model validation workflows at asset management firms.

---
## 2. Solution Overview & Technology Stack

## Strategies 

Because return estimates are noisy, this project focused on risk-based and regularized portfolio optimization and  all methods were evaluated using rolling out-of-sample backtests.

| Strategy | Estimation inputs | Purpose | Strength | Weakness | 
|---|---|--- |---|---|
| Equal Weight | None — 1/N baseline | Baseline | Very robust baseline |Ignores risk structure |
| Minimum Variance | Covariance only | Risk control | Good downside protection | Sensitive to covariance estimation |
| Risk Parity | Covariance only | Robust allocation | Diversified risk exposure | Requires stable covariance |
| Regularised Max Sharpe | Mean + Covariance, L2 penalty | Active tilt | Strong theoretical foundation | Unstable with noisy returns |

## Metrics

Each strategy is evaluated across return, risk, and efficiency dimensions:

| Return Quality | Risk |
|---|---|
| Annualised Return (CAGR) | Annualised Volatility |
| Sharpe Ratio | Maximum Drawdown |
| Sortino Ratio | CVaR (95%) |
| Calmar Ratio | — |

**Turnover** is tracked separately per rebalance — higher turnover means higher transaction costs in production.

To mitigate estimation error in portfolio inputs, the framework optionally applies shrinkage to both expected returns and the covariance matrix. This allows to explore how different levels of regularization affect portfolio stability and out-of-sample performance.

Additionally, each rebalance records **predicted vs realised** Sharpe, Sortino, and Volatility — measuring how accurately each strategy anticipated its own risk-adjusted performance.


---
## Methodology

The framework employs a rolling walk-forward backtesting procedure: portfolio parameters are estimated on a fixed-length historical window, optimized weights are applied to the next rebalancing period, and the window is then advanced to simulate sequential real-time portfolio management.
``` mermaid
flowchart LR
    A["Select rolling training window<br>(1–10 years)"] --> B["Estimate expected returns<br>and covariance matrix"]
    B --> C["Compute portfolio weights<br>for each strategy"]
    C --> D["Apply weights to<br>next rebalance period"]
    D --> E["Record realized returns<br>and risk metrics"]
    E --> F{Next rebalance<br>date?}
    F -->|Yes| G["Slide training window forward"]
    G --> B
    F -->|No| H["Aggregate results<br>and compare strategies"]
```

**Training** — estimate expected returns and the covariance matrix from the rolling window.

**Weights** — each strategy computes portfolio allocations independently using the same inputs.

**Test** — the weights are held fixed until the next rebalance. No adjustments, no look-ahead.

**Window Update** — the training window slides forward by one month, and the process repeats through the entire out-of-sample period.

---
## Data
Daily adjusted price data is downloaded from **Yahoo Finance** using the Python library **yfinance**.

The pipeline performs:

- price alignment across assets

- return calculation

- missing value handling

---
## Reproducibility

All results in this project are fully reproducible.
The workflow follows a deterministic pipeline:
```mermaid
flowchart LR
    A[Download Prices\<br>yfinance] --> B[Compute Returns\<br>Align Series]
    B --> C[Walk-Forward\<br>Backtest]
    C --> D[Evaluation\<br>Metrics]
    D --> E[Streamlit\<br>Dashboard]
```
The repository includes:
- Automated tests  `pytest`
- Continuous integration - GitHub Actions
- Deterministic backtesting - Fixed pipeline 

This ensures the results can be independently verified and extended.

---
## Stack
`Python 3.11` · `scipy` · `scikit-learn` · `pandas` · `numpy` · `yfinance` · `Streamlit` · `Plotly` · `pytest` · `GitHub Actions`

---- 

## Project structure

```
└── 📁portfolio_optimization
    └── 📁.github
        └── 📁workflows
            ├── ci.yaml
    └── 📁.streamlit
        ├── config.toml
    └── 📁src
        └── 📁portfolio_optimization
            ├── __init__.py
            ├── backtest.py
            ├── config.py
            ├── data.py
            ├── main.py
            ├── metrics.py
            ├── optimization.py
    └── 📁tests
        ├── __init__.py
        ├── test_backtest.py
        ├── test_data.py
        ├── test_metrics.py
        ├── test_optimization.py
    ├── .gitignore
    ├── app.py
    ├── pyproject.toml
    ├── README.md
    └── requirements.txt
```
---
## Quickstart

```bash
git clone https://github.com/marieltv/portfolio_optimization.git
cd portfolio_optimization
pip install -e ".[dev]"
streamlit run app.py
```

[Live demo →](https://optimaalinenportfolio.streamlit.app/)

---

## Tests

```bash
pytest -v
```
---

## Results

Evaluated on US defence equities (`BA, NOC, LMT, RTX, AXON, GD`), `2018–2026`, `4-year` rolling training window, `monthly rebalancing`.

| Strategy | CAGR | Sharpe | Sortino | Max Drawdown |
|---|---|---|---|---|
| Equal Weight | 20.3% | 1.05 | 1.08 | -17.5% |
| Min Variance | 19.8% | 1.05 | 1.10 | -16.8% |
| Risk Parity | 20.0% | 1.07 | 1.11 | -15.8% |
| Reg Max Sharpe | 24.9% | 1.16 | 1.23 | -17.4% |

**Regularised Max Sharpe** outperforms on all metrics despite being theoretically the most sensitive to estimation error — the L2 penalty appears sufficient to
control weight concentration in this universe.

**Estimation accuracy** — volatility is predicted with MAE ~0.07 across all strategies, confirming that covariance is more stable than mean returns.
However, Spearman correlations between predicted and actual Sharpe and Sortino are near zero for all strategies, confirming that month-to-month
risk-adjusted returns are not predictable from historical estimates alone — consistent with Merton (1980).

**Ledoit-Wolf and James-Stein shrinkages** improved results marginally (differences of 0.01 or less), suggesting estimation error is not the primary source of out-of-sample degradation in this universe.

**Sharpe and Sortino predictions** are flat and near-constant — a direct consequence of estimating expected returns from a multi-year rolling window. This is not a modelling failure. It reflects a fundamental property of equity returns: covariance is forecastable, mean returns are not. The value of these strategies lies in risk budgeting and drawdown control, not in predicting next month's risk-adjusted return.

---
## Interactive Dashboard

An interactive dashboard built with **Streamlit** allows exploration of the backtest results and strategy behaviour.

The dashboard enables users to:

- select the training window length used for walk-forward optimisation

- compare portfolio strategies across multiple performance metrics

- analyse cumulative returns and drawdowns

- inspect predicted vs realised Sharpe, Sortino, and volatility
- control mean return shrinkage and covariance matrix shrinkage

- evaluate portfolio turnover and allocation dynamics

Interactive charts are rendered using **Plotly**.
