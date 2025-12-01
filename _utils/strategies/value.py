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

def value(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict, # not used
    asset_class_groupings:dict,
    horizon_years: int,
    smooth: int = 32,
    scalar: float = 7.27,
) -> pd.Series:
    """
    Compute the value-style forecast for an instrument based on its relative price
    within its asset class.

    The relative price normaliza o ativo pelo desvio padrão e depois compara
    seu nível atual vs. seus pares (dentro da classe), medido pelo valor
    relativo calculado previamente. Em seguida, o forecast é baseado na
    variação desse valor ao longo de `horizon_years`.

    Parameters
    ----------
    instrument_code : str
        Identifier of the instrument.
    adjusted_prices_dict : dict
        Mapping {instrument_code: pd.Series} of adjusted price series.
    std_dev_dict : dict
        Mapping {instrument_code: standardDeviation} used to compute relative prices.
    carry_prices_dict : dict
        Unused (for compatibility with generic forecast function signature).
    horizon_years : int
        Lookback horizon (in years) used to evaluate relative outperformance.
    smooth : int, default 32
        Smoothing span for exponential moving average.
    scalar : float, default 7.27
        Scaling factor applied to the smoothed forecast.

    Returns
    -------
    pd.Series
        Value forecast series for the instrument.
    """
    relative_price_dict = calculate_relative_price_dict(
        std_dev_dict=std_dev_dict,
        adjusted_prices_dict=adjusted_prices_dict,
        asset_class_groupings=asset_class_groupings,
    )

    value_forecast = calculate_forecast_for_value(
        relative_price=relative_price_dict[instrument_code],
        horizon_years=horizon_years,
        scalar=scalar,
        smooth=smooth,
    )

    return value_forecast


def calculate_forecast_for_value(
    relative_price: pd.Series,
    horizon_years: int,
    smooth: int = 32,
    scalar: float = 7.27,
) -> pd.Series:
    """
    Compute the value forecast based on multi-year relative price mean reversion.

    Method:
        1. Compute `horizon_days = BUSINESS_DAYS_IN_YEAR * horizon_years`.
        2. Compute multi-year outperformance:
               outperformance = (current_rel_price - rel_price_shifted) / horizon_days
        3. Forecast = -outperformance  (mean reversion assumption).
        4. Smooth with EWM (span=`smooth`).
        5. Multiply by `scalar`.

    Parameters
    ----------
    relative_price : pd.Series
        Time series of relative price for an instrument vs. its asset class.
    horizon_years : int
        Lookback window in years.
    smooth : int, default 32
        Span for exponential smoothing.
    scalar : float, default 7.27
        Scaling applied to the smoothed forecast.

    Returns
    -------
    pd.Series
        Value forecast series.
    """
    horizon_days = BUSINESS_DAYS_IN_YEAR * horizon_years

    outperformance = (relative_price - relative_price.shift(horizon_days)) / horizon_days
    forecast = -outperformance  # mean reversion intuition
    smoothed_forecast = forecast.ewm(smooth, min_periods=1).mean()

    return smoothed_forecast * scalar
