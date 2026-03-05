"""
Rolling out-of-sample backtesting engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

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
    first_valid_date = returns.index[train_days]  # no rebalancing before training window is full

    for i, rebalance_date in enumerate(rebalance_dates):

        # Skip until we have a full training window
        if rebalance_date < first_valid_date:
            continue

        train_end_loc   = returns.index.searchsorted(rebalance_date, side="right")
        train_start_loc = max(0, train_end_loc - train_days)
        train = returns.iloc[train_start_loc:train_end_loc]

        if len(train) < 20:
            logger.warning("Skipping %s — only %d training rows.", rebalance_date, len(train))
            continue

        # Test window: current rebalance → next rebalance (exclusive)
        next_date    = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index[-1]
        test_end_loc = returns.index.searchsorted(next_date, side="right")
        test = returns.iloc[train_end_loc:test_end_loc]

        if test.empty:
            continue

        # Annualised estimation inputs
        mean_ann = train.mean().values * cfg.trading_days
        cov_ann  = train.cov().values  * cfg.trading_days

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
            # Predicted metrics — computed from training window + weights
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
            # Actual metrics — realised over the test window
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
            # Store everything
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
    # Assemble final results
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
