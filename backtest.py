"""
Rolling out-of-sample backtesting engine.
"""

from typing import Dict, List
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
Rolling out-of-sample backtesting engine.
"""

from typing import Dict, List
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
