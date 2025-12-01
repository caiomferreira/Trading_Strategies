import pandas as pd
import numpy as np
from _utils.core_functions import *

"""
This is a code based on book:
 "Advanced Futures Trading Strategies", by Robert Carver
 https://www.systematicmoney.org/advanced-futures

This code is copyright, Robert Carver 2022.
Shared under https://www.gnu.org/licenses/gpl-3.0.en.html
You may copy, modify, and share this code as long as this header is retained, and you disclose that it has been edited.
This code comes with no warranty, is not guaranteed to be accurate, and the author is not responsible for any losses that may result from it's use.

Results may not match the book exactly as different data may be used
Results may be different from the corresponding spreadsheet as methods may be slightly different
"""


def breakout(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,      # not used
    carry_prices_dict: dict, # not used
    scalar: float = 1.0,
    horizon: int = 10,
) -> pd.Series:
    """
    Compute a breakout-style forecast for a given instrument.

    The forecast measures how far the current price is from the midpoint
    of the rolling (max, min) breakout channel, scaled into Carver-style
    forecast units and optionally multiplied by a user-defined scalar.

    Parameters
    ----------
    instrument_code : str
        Identifier of the instrument.
    adjusted_prices_dict : dict
        Mapping {instrument_code: pd.Series} containing adjusted prices.
    std_dev_dict : dict
        Unused. Present for API compatibility with other forecast functions.
    carry_prices_dict : dict
        Unused. Present for API compatibility with other forecast functions.
    scalar : float, default 1.0
        Multiplier applied to the forecast.
    horizon : int, default 10
        Lookback window for breakout calculation.

    Returns
    -------
    pd.Series
        Breakout forecast series scaled by `scalar`.
    """
    breakout_forecast = calculate_forecast_for_breakout(
        adjusted_price=adjusted_prices_dict[instrument_code],
        horizon=horizon,
        scalar=scalar,
    )
    return breakout_forecast


def calculate_forecast_for_breakout(
    adjusted_price: pd.Series,
    horizon: int = 10,
    scalar: float = 1.0,
) -> pd.Series:
    """
    Compute the breakout forecast based on the rolling max/min price channel.

    Method:
        1. Compute rolling highest high and lowest low over `horizon`.
        2. Midpoint = (max + min) / 2.
        3. Raw forecast = 40 * (price - midpoint) / (max - min).
        4. Smooth using EWM with span = horizon/4.
        5. Multiply by `scalar`.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series for the instrument.
    horizon : int, default 10
        Rolling window used to compute breakout channel.
    scalar : float, default 1.0
        Multiplier applied to final forecast.

    Returns
    -------
    pd.Series
        Smoothed breakout forecast series.
    """
    max_price = adjusted_price.rolling(horizon, min_periods=1).max()
    min_price = adjusted_price.rolling(horizon, min_periods=1).min()
    mean_price = (max_price + min_price) / 2

    raw_forecast = 40 * (adjusted_price - mean_price) / (max_price - min_price)
    smoothed_forecast = raw_forecast.ewm(span=int(np.ceil(horizon / 4))).mean()

    return smoothed_forecast * scalar
