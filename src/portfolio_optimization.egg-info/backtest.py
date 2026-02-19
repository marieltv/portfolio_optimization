"""
Rolling out-of-sample backtesting engine.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from config import TRAIN_WINDOW_YEARS, TRADING_DAYS
from optimization import (
    equal_weight,
    minimum_variance,
    risk_parity,
    regularized_max_sharpe,
)


def rolling_backtest(
    returns: pd.DataFrame
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame]]:
    """
    Rolling train/test backtest with monthly rebalancing.

    Returns:
        portfolio_returns: daily out-of-sample returns
        weight_history: portfolio weights at each rebalance date
    """

    train_window = TRAIN_WINDOW_YEARS * TRADING_DAYS

    portfolio_returns = {
        "EqualWeight": [],
        "MinVariance": [],
        "RiskParity": [],
        "RegMaxSharpe": [],
    }

    weight_history = {
        "EqualWeight": [],
        "MinVariance": [],
        "RiskParity": [],
        "RegMaxSharpe": [],
    }

    rebalance_dates = returns.index[train_window::21]  # approx monthly

    for rebalance_date in rebalance_dates:

        train_end = rebalance_date
        train_start = returns.index.get_loc(train_end) - train_window

        train = returns.iloc[train_start:returns.index.get_loc(train_end)]
        test = returns.loc[rebalance_date:]

        mean = train.mean().values * TRADING_DAYS
        cov = train.cov().values * TRADING_DAYS

        weights = {
            "EqualWeight": equal_weight(train.shape[1]),
            "MinVariance": minimum_variance(cov),
            "RiskParity": risk_parity(cov),
            "RegMaxSharpe": regularized_max_sharpe(mean, cov),
        }

        # Apply weights for next ~21 trading days
        test_period = test.iloc[:21]

        for name, w in weights.items():
            daily_port_returns = test_period.values @ w
            portfolio_returns[name].extend(daily_port_returns)
            weight_history[name].append(w)

    # Convert to pandas
    portfolio_returns = {
        name: pd.Series(values, index=returns.index[train_window:train_window+len(values)])
        for name, values in portfolio_returns.items()
    }

    weight_history = {
        name: pd.DataFrame(values, columns=returns.columns)
        for name, values in weight_history.items()
    }

    return portfolio_returns, weight_history

"""
Rolling out-of-sample backtesting engine.


from typing import Dict, List
import pandas as pd
import numpy as np

from srs.config import TRAIN_WINDOW_YEARS, TRADING_DAYS
from optimization import (
    equal_weight,
    minimum_variance,
    risk_parity,
    regularized_max_sharpe,
)


def rolling_backtest(
    returns: pd.DataFrame
) -> Dict[str, pd.Series]:

    strategies = {
        "EqualWeight": [],
        "MinVariance": [],
        "RiskParity": [],
        "RegMaxSharpe": [],
    }

    train_window = TRAIN_WINDOW_YEARS * TRADING_DAYS
    dates: List[pd.Timestamp] = []

    for i in range(train_window, len(returns)):
        train = returns.iloc[i - train_window:i]
        test = returns.iloc[i]

        mean = train.mean().values * TRADING_DAYS
        cov = train.cov().values * TRADING_DAYS

        weights = {
            "EqualWeight": equal_weight(train.shape[1]),
            "MinVariance": minimum_variance(cov),
            "RiskParity": risk_parity(cov),
            "RegMaxSharpe": regularized_max_sharpe(mean, cov),
        }

        for name, w in weights.items():
            strategies[name].append(test.values @ w)

        dates.append(test.name)

    return {
        name: pd.Series(values, index=dates)
        for name, values in strategies.items()
    }
"""