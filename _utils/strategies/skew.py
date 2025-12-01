import pandas as pd
import numpy as np
from _utils.core_functions import *

from _utils.strategies.trend_asset_classes import (
    calculate_asset_class_price_dict,
    calculate_relative_price_dict
    )


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

def skew(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,      # not used except for instrument_risk passthrough
    carry_prices_dict: dict, # not used
    horizon: int = 60,
    scalar: float = 33.3,
) -> pd.Series:
    """
    Compute a skew-based forecast using rolling skewness of percentage returns.

    The idea is that negative skew (frequent small gains, occasional large losses)
    and positive skew (frequent small losses, occasional large gains) may provide
    predictive information about reversals or continuation depending on the signal
    design. Here, the forecast uses:
        - rolling skew of percent returns over `horizon`
        - inversion (negative sign) to align with mean-reversion intuition
        - exponential smoothing
        - scaling and capping

    Parameters
    ----------
    instrument_code : str
        Identifier of the instrument.
    adjusted_prices_dict : dict
        Mapping {instrument_code: pd.Series} with adjusted prices.
    std_dev_dict : dict
        Mapping {instrument_code: standardDeviation}, whose `.current_price`
        is used to compute percentage returns.
    carry_prices_dict : dict
        Unused. Present for compatibility with other rule-based forecast APIs.
    horizon : int, default 60
        Rolling window length (in days) used to compute skewness.
    scalar : float, default 33.3
        Scaling applied to the smoothed skew forecast.

    Returns
    -------
    pd.Series
        Skew-based forecast, capped to [-20, 20].
    """
    skew_forecast = calculate_forecast_for_skew(
        adjusted_price=adjusted_prices_dict[instrument_code],
        instrument_risk=std_dev_dict[instrument_code],
        scalar=scalar,
        horizon=horizon,
    )

    return skew_forecast


def calculate_forecast_for_skew(
    adjusted_price: pd.Series,
    instrument_risk: standardDeviation,
    scalar: float,
    horizon: int = 60,
) -> pd.Series:
    """
    Compute a skewness-based forecast using rolling skew of percent returns.

    Method:
        1. Compute percentage returns using:
               (price_t - price_{t-1}) / instrument_risk.current_price.shift(1)
           (Using current_price from risk object for normalization.)
        2. Compute rolling skewness over `horizon`.
        3. Multiply by -1 to invert the signal (mean-reversion assumption).
        4. Smooth using EWM with span = horizon / 4.
        5. Scale by `scalar`.
        6. Clip to [-20, 20].

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series of the instrument.
    instrument_risk : standardDeviation
        Object containing volatility info and `.current_price`.
    scalar : float
        Scaling factor applied after smoothing.
    horizon : int, default 60
        Rolling window length for computing skewness.

    Returns
    -------
    pd.Series
        Skew-based forecast, already capped to [-20, 20].
    """
    current_price = instrument_risk.current_price

    perc_returns = adjusted_price.diff() / current_price.shift(1)
    raw_forecast = -perc_returns.rolling(horizon).skew()

    smoothed_forecast = raw_forecast.ewm(
        span=int(horizon / 4),
        min_periods=1
    ).mean()

    scaled_forecast = smoothed_forecast * scalar
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast
