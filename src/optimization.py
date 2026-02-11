"""
Portfolio optimization methods.
"""

import numpy as np
from scipy.optimize import minimize
from config import MAX_WEIGHT, L2_PENALTY


def equal_weight(n_assets: int) -> np.ndarray:
    return np.ones(n_assets) / n_assets


def minimum_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]

    def objective(w: np.ndarray) -> float:
        return w.T @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, MAX_WEIGHT)] * n
    init = equal_weight(n)

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    w = result.x
    return w / np.sum(w)


def risk_parity(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]

    def portfolio_variance(w: np.ndarray) -> float:
        return w.T @ cov @ w

    def risk_contribution(w: np.ndarray) -> np.ndarray:
        total_var = portfolio_variance(w)
        marginal = cov @ w
        return w * marginal / total_var

    def objective(w: np.ndarray) -> float:
        rc = risk_contribution(w)
        target = np.ones(n) / n
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, MAX_WEIGHT)] * n
    init = equal_weight(n)

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    w = result.x
    return w / np.sum(w)


def regularized_max_sharpe(
    mean: np.ndarray,
    cov: np.ndarray
) -> np.ndarray:
    n = len(mean)

    def objective(w: np.ndarray) -> float:
        ret = w @ mean
        vol = np.sqrt(w.T @ cov @ w)
        penalty = L2_PENALTY * np.sum(w**2)
        return -(ret / vol) + penalty

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, MAX_WEIGHT)] * n
    init = equal_weight(n)

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    w = result.x
    return w / np.sum(w)
