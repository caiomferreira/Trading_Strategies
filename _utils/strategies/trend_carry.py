import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm
from copy import copy
from scipy.interpolate import interp1d


from _utils.core_functions import *
from _utils.portfoliohandcrafiting import *
from _utils.strategies.trend_simple_filter import *
from _utils.strategies.simple_carry import *

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
FDM_DICT = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}

def calculate_position_dict_with_multiple_trend_forecast_applied_and_adjustment(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    fast_spans: list,
) -> dict:
    """
    Build a dictionary of positions for multiple instruments using a multi-trend EWMAC forecast.

    Parameters
    ----------
    adjusted_prices_dict : dict
        Mapping {instrument_code: adjusted_price_series}.
    average_position_contracts_dict : dict
        Mapping {instrument_code: average_position_series}.
    std_dev_dict : dict
        Mapping {instrument_code: standardDeviation object}.
    fast_spans : list
        List of fast spans for EWMAC calculations.

    Returns
    -------
    dict
        Mapping {instrument_code: position_series}.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = dict(
        [
            (
                instrument_code,
                calculate_position_with_multiple_trend_forecast_applied_and_adjustment(
                    adjusted_prices_dict[instrument_code],
                    average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    fast_spans=fast_spans,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_trend_filter


def calculate_position_with_multiple_trend_forecast_applied_and_adjustment(
    adjusted_price: pd.Series,
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_spans: list,
) -> pd.Series:
    """
    Compute position series for one instrument using multiple EWMAC trend forecasts.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series.
    average_position : pd.Series
        Average position profile in contracts.
    stdev_ann_perc : standardDeviation
        Standard deviation object providing daily volatility estimates.
    fast_spans : list
        Fast spans for multiple EWMAC rules.

    Returns
    -------
    pd.Series
        Position series scaled by forecast and average position.
    """
    forecast = calculate_combined_ewmac_forecast_and_adjustment(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_spans=fast_spans,
    )

    return forecast * average_position / 10


def calculate_combined_ewmac_forecast_and_adjustment(
    adjusted_price: pd.Series, stdev_ann_perc: standardDeviation, fast_spans: list
) -> pd.Series:
    """
    Combine multiple EWMAC forecasts, apply diversification multiplier (FDM),
    cap values, and return a single aggregated forecast.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series.
    stdev_ann_perc : standardDeviation
        Standard deviation object providing daily volatility estimates.
    fast_spans : list
        List of fast spans for EWMAC rules.

    Returns
    -------
    pd.Series
        Combined and adjusted forecast series capped to [-20, 20].
    """
    all_forecasts_as_list = [
        calculate_forecast_for_ewmac_and_adjustment(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            fast_span=fast_span,
        )
        for fast_span in fast_spans
    ]

    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    rule_count = len(fast_spans)
    fdm = FDM_DICT[rule_count]

    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast


def calculate_forecast_for_ewmac_and_adjustment(
    adjusted_price: pd.Series, stdev_ann_perc: standardDeviation, fast_span: int = 64
):
    """
    Compute an individual EWMAC forecast with volatility adjustment and span-specific scaling.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series.
    stdev_ann_perc : standardDeviation
        Standard deviation object providing daily volatility terms.
    fast_span : int, optional
        Fast span for EWMAC (default=64).

    Returns
    -------
    pd.Series
        Forecast series after volatility adjustment and transformation.
    """
    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    ewmac_values = ewmac(adjusted_price, fast_span=fast_span, slow_span=fast_span * 4)
    daily_price_vol = stdev_ann_perc.daily_risk_price_terms()
    risk_adjusted_ewmac = ewmac_values / daily_price_vol
    forecast_scalar = scalar_dict[fast_span]
    scaled_ewmac = risk_adjusted_ewmac * forecast_scalar

    if fast_span == 2:
        capped_ewmac = double_v(scaled_ewmac)
    elif fast_span == 4 or fast_span == 64:
        capped_ewmac = scale_and_cap(scaled_ewmac)
    else:
        capped_ewmac = scaled_ewmac.clip(-20, 20)

    return capped_ewmac


def double_v(scaled_forecast: pd.Series) -> pd.Series:
    """
    Apply a nonlinear 'double-V' transformation to extreme forecasts
    for very fast spans (e.g., fast_span=2).

    Parameters
    ----------
    scaled_forecast : pd.Series
        EWMAC forecast already scaled by span-specific factor.

    Returns
    -------
    pd.Series
        Nonlinearly transformed forecast.
    """
    new_forecast = copy(scaled_forecast)
    new_forecast[scaled_forecast > 20] = 0
    new_forecast[scaled_forecast < -20] = 0

    new_forecast[(scaled_forecast >= -20) & (scaled_forecast < -10)] = (
        new_forecast[(scaled_forecast >= -20) & (scaled_forecast < -10)] * -2
    ) - 40

    new_forecast[(scaled_forecast >= -10) & (scaled_forecast < 10)] = (
        new_forecast[(scaled_forecast >= -10) & (scaled_forecast < +10)] * 2
    )

    new_forecast[(scaled_forecast >= 10) & (scaled_forecast < 20)] = (
        new_forecast[(scaled_forecast >= 10) & (scaled_forecast < +20)] * -2
    ) + 40

    return new_forecast


def scale_and_cap(scaled_forecast: pd.Series) -> pd.Series:
    """
    Apply a simple linear scaling and capping transformation
    for specific spans (fast_span = 4 or 64).

    Parameters
    ----------
    scaled_forecast : pd.Series
        Forecast series before scaling.

    Returns
    -------
    pd.Series
        Rescaled and capped forecast series.
    """
    rescaled_forecast = 1.25 * scaled_forecast
    capped_forecast = rescaled_forecast.clip(-1.25 * 15, 1.25 * 15)
    return capped_forecast
