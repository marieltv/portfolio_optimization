"""
Portfolio optimisation strategies.

Design: each strategy is a *callable class* that implements the
``Optimizer`` protocol.  This lets you swap strategies without changing
the backtest loop, and makes it trivial to add new ones.

Strategies
----------
EqualWeight         — 1/N benchmark, no estimation required.
MinimumVariance     — minimise σ²_p subject to weight constraints.
RiskParity          — equalise marginal risk contributions.
RegularisedMaxSharpe — maximise Sharpe with L2 weight regularisation.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import minimize, OptimizeResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def _check_cov(cov: np.ndarray) -> None:
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be a square 2-D array, got shape {cov.shape}")
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("Covariance matrix is not symmetric.")


def _check_result(result: OptimizeResult, name: str) -> None:
    if not result.success:
        raise OptimizationError(name, result.message)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class OptimizationError(Exception):
    """Raised when a scipy optimiser fails to converge."""
    def __init__(self, method: str, message: str) -> None:
        super().__init__(f"[{method}] optimisation failed: {message}")


# ---------------------------------------------------------------------------
# Protocol — every optimiser must satisfy this interface
# ---------------------------------------------------------------------------

@runtime_checkable
class Optimizer(Protocol):
    """Callable that maps estimation inputs → portfolio weight vector."""
    name: str

    def __call__(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        mean : annualised expected return vector, shape (n_assets,).
        cov  : annualised covariance matrix, shape (n_assets, n_assets).

        Returns
        -------
        np.ndarray
            Weight vector that sums to 1, shape (n_assets,).
        """
        ...


# ---------------------------------------------------------------------------
# Concrete optimisers
# ---------------------------------------------------------------------------

class EqualWeight:
    """1/N benchmark — no estimation required."""

    name = "EqualWeight"

    def __call__(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        return _equal_weight(len(mean))


class MinimumVariance:
    """
    Minimum-variance portfolio.

    Solves:
        min   w' Σ w
        s.t.  Σ wᵢ = 1,  0 ≤ wᵢ ≤ max_weight
    """

    name = "MinVariance"

    def __init__(self, max_weight: float = 0.30) -> None:
        self.max_weight = max_weight

    def __call__(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        _check_cov(cov)
        n = cov.shape[0]

        result = minimize(
            fun=lambda w: float(w @ cov @ w),
            x0=_equal_weight(n),
            method="SLSQP",
            bounds=[(0.0, self.max_weight)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        _check_result(result, self.name)
        w = result.x
        return w / w.sum()


class RiskParity:
    """
    Risk-parity (equal risk contribution) portfolio.

    Solves:
        min   Σᵢ Σⱼ (RCᵢ - RCⱼ)²
        s.t.  Σ wᵢ = 1,  0 ≤ wᵢ ≤ max_weight

    where RC_i = w_i (Σw)_i / (w'Σw) is asset i's fractional risk
    contribution.
    """

    name = "RiskParity"

    def __init__(self, max_weight: float = 0.30) -> None:
        self.max_weight = max_weight

    def __call__(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        _check_cov(cov)
        n = cov.shape[0]
        target_rc = np.ones(n) / n

        def objective(w: np.ndarray) -> float:
            port_var = max(float(w @ cov @ w), 1e-12)
            rc = w * (cov @ w) / port_var
            return float(np.sum((rc - target_rc) ** 2))

        result = minimize(
            fun=objective,
            x0=_equal_weight(n),
            method="SLSQP",
            bounds=[(0.0, self.max_weight)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        _check_result(result, self.name)
        w = result.x
        return w / w.sum()


class RegularisedMaxSharpe:
    """
    L2-regularised maximum-Sharpe portfolio.

    Solves:
        max   (w'μ - rf) / σ_p  −  λ ‖w‖²
        s.t.  Σ wᵢ = 1,  0 ≤ wᵢ ≤ max_weight

    The L2 penalty (λ) shrinks weights toward equal-weight, reducing
    sensitivity to return-estimate noise.

    Parameters
    ----------
    risk_free_rate : annualised risk-free rate (same units as ``mean``).
    max_weight : per-asset weight upper bound.
    l2_penalty : regularisation strength λ.
    """

    name = "RegMaxSharpe"

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        max_weight: float = 0.30,
        l2_penalty: float = 0.1,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.l2_penalty = l2_penalty

    def __call__(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        _check_cov(cov)
        n = len(mean)

        def objective(w: np.ndarray) -> float:
            excess = float(w @ mean) - self.risk_free_rate
            vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
            penalty = self.l2_penalty * float(np.sum(w ** 2))
            return -(excess / vol) + penalty

        result = minimize(
            fun=objective,
            x0=_equal_weight(n),
            method="SLSQP",
            bounds=[(0.0, self.max_weight)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        _check_result(result, self.name)
        w = result.x
        return w / w.sum()


# ---------------------------------------------------------------------------
# Default strategy registry — ordered for display consistency
# ---------------------------------------------------------------------------

def default_optimizers(cfg) -> list[Optimizer]:
    """
    Return the four standard strategies initialised from a ``BacktestConfig``.
    """
    return [
        EqualWeight(),
        MinimumVariance(max_weight=cfg.max_weight),
        RiskParity(max_weight=cfg.max_weight),
        RegularisedMaxSharpe(
            risk_free_rate=cfg.risk_free_rate,
            max_weight=cfg.max_weight,
            l2_penalty=cfg.l2_penalty,
        ),
    ]
