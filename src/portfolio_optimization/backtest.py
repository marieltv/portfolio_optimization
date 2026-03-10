"""
Rolling out-of-sample backtesting engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio_optimization.config import BacktestConfig, DEFAULT_CONFIG
from portfolio_optimization.optimization import (
    Optimizer,
    EqualWeight,
    OptimizationError,
    default_optimizers,
)

logger = logging.getLogger(__name__)

_FALLBACK = EqualWeight()


# ---------------------------------------------------------------------------
# Shrinkage estimators
# ---------------------------------------------------------------------------

def _ledoit_wolf_cov(train: pd.DataFrame, trading_days: int) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage estimator for the covariance matrix.
    Shrinks the sample covariance toward a structured target,
    reducing estimation error in small samples.
    """
    lw = LedoitWolf().fit(train.values)
    return lw.covariance_ * trading_days


def _james_stein_mean(train: pd.DataFrame, trading_days: int) -> np.ndarray:
    """
    James-Stein shrinkage estimator for the mean return vector.
    Shrinks individual asset means toward the grand mean,
    reducing the impact of extreme return estimates.
    """
    mu = train.mean().values * trading_days
    n = len(mu)

    if n <= 2:
        return mu

    grand_mean = mu.mean()
    mu_centered = mu - grand_mean
    norm_sq = float(mu_centered @ mu_centered)

    if norm_sq < 1e-12:
        return mu

    sigma_sq = float(train.var().mean()) * trading_days / len(train)
    alpha = min((n - 2) * sigma_sq / (n * norm_sq), 1.0)

    return grand_mean + (1 - alpha) * mu_centered


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Container for all outputs of a rolling backtest run.

    Attributes
    ----------
    returns   : daily out-of-sample portfolio returns per strategy.
    weights   : weight at each rebalance date per strategy.
    turnover  : one-way turnover at each rebalance per strategy.
    predicted : predicted vs actual Sharpe, Sortino, Volatility per rebalance.
    """
    returns:   Dict[str, pd.Series]    = field(default_factory=dict)
    weights:   Dict[str, pd.DataFrame] = field(default_factory=dict)
    turnover:  Dict[str, pd.Series]    = field(default_factory=dict)
    predicted: Dict[str, pd.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def rolling_backtest(
    returns: pd.DataFrame,
    optimizers: List[Optimizer] | None = None,
    cfg: BacktestConfig = DEFAULT_CONFIG,
) -> BacktestResult:
    """
    Walk-forward backtest with configurable rebalancing frequency.

    Parameters
    ----------
    returns    : daily asset returns, shape (T, N), with a DatetimeIndex.
    optimizers : list of Optimizer instances to evaluate.
    cfg        : backtest configuration object.

    Returns
    -------
    BacktestResult
    """
    if optimizers is None:
        optimizers = default_optimizers(cfg)

    names = [opt.name for opt in optimizers]
    result = BacktestResult(
        returns={n: []  for n in names},
        weights={n: []  for n in names},
        turnover={n: [] for n in names},
        predicted={n: [] for n in names},
    )
    prev_weights: Dict[str, np.ndarray | None] = {n: None for n in names}
    rebalance_index: List[pd.Timestamp] = []

    train_days = cfg.train_window_days
    daily_rf   = cfg.risk_free_rate / cfg.trading_days

    rebalance_dates  = returns.resample(cfg.rebalance_frequency).last().index
    first_valid_date = returns.index[train_days]

    for i, rebalance_date in enumerate(rebalance_dates):

        if rebalance_date < first_valid_date:
            continue

        train_end_loc   = returns.index.searchsorted(rebalance_date, side="right")
        train_start_loc = max(0, train_end_loc - train_days)
        train = returns.iloc[train_start_loc:train_end_loc]

        if len(train) < 20:
            logger.warning("Skipping %s — only %d training rows.", rebalance_date, len(train))
            continue

        next_date    = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index[-1]
        test_end_loc = returns.index.searchsorted(next_date, side="right")
        test = returns.iloc[train_end_loc:test_end_loc]

        if test.empty:
            continue

        # ----------------------------------------------------------------
        # Estimation — apply shrinkage if configured
        # ----------------------------------------------------------------
        cov_ann  = _ledoit_wolf_cov(train, cfg.trading_days) if cfg.use_ledoit_wolf \
                   else train.cov().values * cfg.trading_days

        mean_ann = _james_stein_mean(train, cfg.trading_days) if cfg.use_james_stein \
                   else train.mean().values * cfg.trading_days

        rebalance_index.append(rebalance_date)

        for opt in optimizers:
            name = opt.name
            try:
                w = opt(mean_ann, cov_ann)
            except OptimizationError as exc:
                logger.warning(
                    "Falling back to EqualWeight for %s at %s: %s", name, rebalance_date, exc
                )
                w = _FALLBACK(mean_ann, cov_ann)

            # ----------------------------------------------------------------
            # Predicted metrics
            # ----------------------------------------------------------------
            pred_ret = float(w @ mean_ann)
            pred_vol = float(np.sqrt(max(w @ cov_ann @ w, 1e-12)))
            pred_sharpe = (pred_ret - cfg.risk_free_rate) / pred_vol

            train_port_rets = train.values @ w
            downside_train  = train_port_rets[train_port_rets < daily_rf] - daily_rf
            pred_downside_vol = (
                float(np.sqrt((downside_train ** 2).mean()) * np.sqrt(cfg.trading_days))
                if len(downside_train) > 0 else np.nan
            )
            pred_sortino = (
                (pred_ret - cfg.risk_free_rate) / pred_downside_vol
                if pred_downside_vol and pred_downside_vol > 1e-12 else np.nan
            )

            # ----------------------------------------------------------------
            # Actual metrics
            # ----------------------------------------------------------------
            port_returns = pd.Series(test.values @ w, index=test.index)

            actual_vol = float(port_returns.std() * np.sqrt(cfg.trading_days))
            actual_ret = float(port_returns.mean() * cfg.trading_days)
            actual_sharpe = (
                (actual_ret - cfg.risk_free_rate) / actual_vol
                if actual_vol > 1e-12 else np.nan
            )

            downside_actual = port_returns[port_returns < daily_rf] - daily_rf
            actual_downside_vol = (
                float(np.sqrt((downside_actual ** 2).mean()) * np.sqrt(cfg.trading_days))
                if len(downside_actual) > 0 else np.nan
            )
            actual_sortino = (
                (actual_ret - cfg.risk_free_rate) / actual_downside_vol
                if actual_downside_vol and actual_downside_vol > 1e-12 else np.nan
            )

            # ----------------------------------------------------------------
            # Store
            # ----------------------------------------------------------------
            result.returns[name].extend(port_returns.tolist())

            result.weights[name].append(w)
            prev = prev_weights[name]
            result.turnover[name].append(
                float(np.sum(np.abs(w - prev))) if prev is not None else 0.0
            )
            prev_weights[name] = w

            result.predicted[name].append({
                "date":           rebalance_date,
                "pred_sharpe":    pred_sharpe,
                "pred_sortino":   pred_sortino,
                "pred_vol":       pred_vol,
                "actual_sharpe":  actual_sharpe,
                "actual_sortino": actual_sortino,
                "actual_vol":     actual_vol,
            })

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    n_oos     = len(next(iter(result.returns.values())))
    oos_start = rebalance_index[0] if rebalance_index else returns.index[train_days]
    oos_index = returns.loc[oos_start:].index[:n_oos]

    result.returns = {
        name: pd.Series(vals, index=oos_index, name=name)
        for name, vals in result.returns.items()
    }
    result.weights = {
        name: pd.DataFrame(rows, index=rebalance_index[:len(rows)], columns=returns.columns)
        for name, rows in result.weights.items()
    }
    result.turnover = {
        name: pd.Series(vals, index=rebalance_index[:len(vals)], name=name)
        for name, vals in result.turnover.items()
    }
    result.predicted = {
        name: pd.DataFrame(rows).set_index("date")
        for name, rows in result.predicted.items()
    }

    return result