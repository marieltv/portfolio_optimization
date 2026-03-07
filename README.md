# Out-of-Sample Evaluation of Portfolio Optimization Methods Under Estimation Uncertainty

[![CI](https://github.com/marieltv/portfolio_optimization/actions/workflows/ci.yaml/badge.svg)](https://github.com/marieltv/portfolio_optimization/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/live%20demo-streamlit-ff4b4b)](https://portfolio-optimisation.streamlit.app)

## 1. Business Problem

Asset managers and quantitative analysts face a consistent challenge: selecting an allocation strategy that is robust out-of-sample, not just historically attractive. Mean-variance optimisation has been the industry standard since Markowitz (1952), yet in practice it is well-documented to be highly sensitive to estimation error — small changes in expected return inputs produce dramatically different, often extreme portfolios.

This project addresses two operational questions directly relevant to portfolio management:

1. **Which allocation strategy delivers the best risk-adjusted return when deployed forward in time** — not fitted to history?
2. **How accurately does each strategy predict its own performance** — and which strategies should be trusted when their assumptions are violated by regime change?

The predicted vs actual framework makes estimation error visible and measurable, rather than burying it in aggregate backtest statistics. This is directly applicable to strategy selection, risk budgeting, and model validation workflows at asset management firms.

---
## 2. Solution Overview & Technology Stack

### Stratigies 

### Methodology

The evaluation follows a walk-forward backtesting framework designed to simulate realistic portfolio management.

#### Training phase

- Historical asset returns are used to estimate expected returns and covariance matrix.

- Portfolio weights are computed using each optimization strategy.

#### Testing phase

- The portfolio is held for a fixed rebalancing period.

- Realized returns and risk metrics are recorded.

#### Walk-forward process

- Train model on historical window

- Compute optimal portfolio weights

- Apply weights to next out-of-sample period

- Expand training window

- Repeat

This avoids look-ahead bias and evaluates strategies under realistic deployment conditions.

---
#### Stack
`Python 3.11` · `scipy` · `pandas` · `numpy` · `yfinance` · `Streamlit` · `Plotly` · `pytest` · `GitHub Actions`

---- 

#### Project folder

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
    ├── 📁venv
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
