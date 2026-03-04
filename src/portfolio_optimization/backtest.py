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
    returns : daily out-of-sample portfolio returns per strategy.
    weights : weight at each rebalance date per strategy.
    turnover : one-way turnover at each rebalance per strategy.
    """
    returns:  Dict[str, pd.Series]    = field(default_factory=dict)
    weights:  Dict[str, pd.DataFrame] = field(default_factory=dict)
    turnover: Dict[str, pd.Series]    = field(default_factory=dict)


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

    The backtest holds weights constant between rebalance dates, mimicking
    realistic portfolio management.  The first ``train_window_days`` rows
    are consumed by the initial training window and do not appear in results.

    Parameters
    ----------
    returns : daily asset returns, shape (T, N), with a DatetimeIndex.
    optimizers : list of Optimizer instances to evaluate.  Defaults to the
                 four standard strategies from ``default_optimizers(cfg)``.
    cfg : configuration object; defaults to ``DEFAULT_CONFIG``.

    Returns
    -------
    BacktestResult
        ``.returns``  — dict[strategy_name, pd.Series of daily returns]
        ``.weights``  — dict[strategy_name, pd.DataFrame of weights at each rebalance]
        ``.turnover`` — dict[strategy_name, pd.Series of turnover at each rebalance]
    """
    if optimizers is None:
        optimizers = default_optimizers(cfg)

    names = [opt.name for opt in optimizers]
    result = BacktestResult(
        returns={n: [] for n in names},
        weights={n: [] for n in names},
        turnover={n: [] for n in names},
    )
    prev_weights: Dict[str, np.ndarray | None] = {n: None for n in names}
    rebalance_index: List[pd.Timestamp] = []

    train_days = cfg.train_window_days
    rebalance_dates = returns.resample(cfg.rebalance_frequency).last().index

    first_valid_date = returns.index[train_days]

    for i, rebalance_date in enumerate(rebalance_dates):
        if rebalance_date < first_valid_date:
            continue
        train_end_loc = returns.index.searchsorted(rebalance_date, side="right")
        train_start_loc = max(0, train_end_loc - train_days)
        train = returns.iloc[train_start_loc:train_end_loc]

        if len(train) < 20:   # skip if insufficient history
            logger.warning("Skipping %s — only %d training rows.", rebalance_date, len(train))
            continue

        # Test window: current rebalance → next rebalance (exclusive)
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index[-1]
        test_start_loc = train_end_loc
        test_end_loc   = returns.index.searchsorted(next_date, side="right")
        test = returns.iloc[test_start_loc:test_end_loc]
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
                logger.warning("Falling back to EqualWeight for %s at %s: %s", name, rebalance_date, exc)
                w = _FALLBACK(mean_ann, cov_ann)

            # Daily portfolio returns for the test window
            port_returns = test.values @ w
            result.returns[name].extend(port_returns.tolist())

            # Weight and turnover tracking
            result.weights[name].append(w)
            prev = prev_weights[name]
            result.turnover[name].append(
                float(np.sum(np.abs(w - prev))) if prev is not None else 0.0
            )
            prev_weights[name] = w

    # -----------------------------------------------------------------------
    # Assemble final result — returns indexed to dates, weights to rebalances
    # -----------------------------------------------------------------------
    # Build a clean date index for the out-of-sample returns
    n_oos = len(next(iter(result.returns.values())))
    # The OOS period starts after the first rebalance date
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

    return result
