import pandas as pd
import numpy as np
from _utils.core_functions import *

from _utils.strategies.trend_simple_filter import calculate_scaled_forecast_for_ewmac

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

def accel(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,  # not used
    fast_span: int = 32,
    scalar: float = 7.27,
) -> pd.Series:
    """
    Compute an acceleration-style forecast based on the change in EWMAC trend strength.

    The idea is:
        - Compute a scaled EWMAC forecast.
        - Measure how much it has accelerated over `fast_span` periods.
        - Scale and cap the result to standard forecast units.

    Parameters
    ----------
    instrument_code : str
        Identifier for the instrument.
    adjusted_prices_dict : dict
        Mapping {instrument_code: pd.Series} with adjusted prices.
    std_dev_dict : dict
        Mapping {instrument_code: standardDeviation} used for EWMAC scaling.
    carry_prices_dict : dict
        Unused (kept for compatibility with generic forecast signature).
    fast_span : int, default 32
        Lookback span used for fast EWMAC computation and lag comparison.
    scalar : float, default 7.27
        Multiplier that scales the acceleration forecast.

    Returns
    -------
    pd.Series
        Acceleration forecast series, already capped to [-20, 20].
    """
    accel_forecast = calculate_forecast_for_accel(
        adjusted_price=adjusted_prices_dict[instrument_code],
        stdev_ann_perc=std_dev_dict[instrument_code],
        scalar=scalar,
        fast_span=fast_span,
    )

    return accel_forecast


def calculate_forecast_for_accel(
    adjusted_price: pd.Series,
    stdev_ann_perc: standardDeviation,
    scalar: float,
    fast_span: int = 64,
) -> pd.Series:
    """
    Compute the acceleration forecast based on EWMAC dynamics.

    Steps:
        1. Compute a scaled EWMAC forecast (Carver-style normalized signal).
        2. Compute its acceleration:
               accel_raw = ewmac_forecast - ewmac_forecast.shift(fast_span)
        3. Multiply by `scalar`.
        4. Clip to [-20, 20].

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series of the instrument.
    stdev_ann_perc : standardDeviation
        Annualized volatility estimator used to standardize EWMAC.
    scalar : float
        Multiplier applied to the acceleration component.
    fast_span : int, default 64
        Span for the fast EWMAC and the lag used to measure acceleration.

    Returns
    -------
    pd.Series
        Acceleration forecast series, capped to [-20, 20].
    """
    ewmac_forecast = calculate_scaled_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )

    accel_raw_forecast = ewmac_forecast - ewmac_forecast.shift(fast_span)
    scaled_accel_forecast = accel_raw_forecast * scalar
    capped_accel_forecast = scaled_accel_forecast.clip(-20, 20)

    return capped_accel_forecast
