# Out-of-Sample Evaluation of Portfolio Optimization Methods Under Estimation Uncertainty

[![CI](https://github.com/marieltv/portfolio_optimization/actions/workflows/ci.yaml/badge.svg)](https://github.com/marieltv/portfolio_optimization/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/live%20demo-streamlit-ff4b4b)](https://portfolio-optimisation.streamlit.app)

## 1. Business Problem

Portfolio managers must allocate capital across assets while balancing return and risk. Traditional allocation approaches often rely on:
- static allocations
- naive diversification
- single in-sample optimization

These approaches frequently overfit historical data and fail when market conditions change.

The key challenge is:

How can we evaluate portfolio construction strategies in a realistic setting that mimics real investment decisions?

This project simulates a real workflow:

Train portfolio models on past data

Rebalance periodically

Evaluate performance on unseen data

This allows comparison of strategies under true out-of-sample conditions.
