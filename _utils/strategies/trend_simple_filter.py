import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm
from copy import copy


from _utils.core_functions import *
from _utils.portfoliohandcrafiting import *

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

# SIGNAL CREATION

import pandas as pd
from copy import copy

def ewmac(adjusted_price: pd.Series, fast_span: int = 64, slow_span: int = 252) -> pd.Series:
    """
    Compute the Exponentially Weighted Moving Average Crossover (EWMAC) indicator.

    Parameters
    ----------
    adjusted_price : pd.Series
        Series of adjusted prices for the instrument.
    fast_span : int, optional
        Span for the fast EWMA (default is 64).
    slow_span : int, optional
        Span for the slow EWMA (default is 252).

    Returns
    -------
    pd.Series
        Difference between fast and slow EWMA (EWMAC values).
    """
    slow_ewma = adjusted_price.ewm(span=slow_span, min_periods=2).mean()
    fast_ewma = adjusted_price.ewm(span=fast_span, min_periods=2).mean()
    return fast_ewma - slow_ewma


def calculate_position_dict_with_trend_filter_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
) -> dict:
    """
    Apply a long-only trend filter (EWMAC > 0) to all instruments in a dictionary.

    Parameters
    ----------
    adjusted_prices_dict : dict
        Dictionary of price Series per instrument.
    average_position_contracts_dict : dict
        Dictionary of average position Series per instrument.

    Returns
    -------
    dict
        Dictionary with filtered position Series where negative EWMAC signals set position to 0.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = {
        instrument_code: calculate_position_with_trend_filter_applied(
            adjusted_prices_dict[instrument_code],
            average_position_contracts_dict[instrument_code],
        )
        for instrument_code in list_of_instruments
    }
    return position_dict_with_trend_filter


def calculate_position_with_trend_filter_applied(
    adjusted_price: pd.Series, average_position: pd.Series
) -> pd.Series:
    """
    Apply a long-only EWMAC trend filter to a single instrument.

    Parameters
    ----------
    adjusted_price : pd.Series
        Price series of the instrument.
    average_position : pd.Series
        Average position series (in contracts or normalized units).

    Returns
    -------
    pd.Series
        Filtered position Series with zero where EWMAC < 0.
    """
    filtered_position = copy(average_position)
    ewmac_values = ewmac(adjusted_price)
    bearish = ewmac_values < 0
    filtered_position[bearish] = 0
    return filtered_position


def calculate_position_dict_with_symmetric_trend_filter_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
) -> dict:
    """
    Apply a symmetric EWMAC trend filter to all instruments (long if EWMAC > 0, short if EWMAC < 0).

    Parameters
    ----------
    adjusted_prices_dict : dict
        Dictionary of price Series per instrument.
    average_position_contracts_dict : dict
        Dictionary of average position Series per instrument.

    Returns
    -------
    dict
        Dictionary of position Series with direction flipped for negative EWMAC.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = {
        instrument_code: calculate_position_with_symmetric_trend_filter_applied(
            adjusted_prices_dict[instrument_code],
            average_position_contracts_dict[instrument_code],
        )
        for instrument_code in list_of_instruments
    }
    return position_dict_with_trend_filter


def calculate_position_with_symmetric_trend_filter_applied(
    adjusted_price: pd.Series, average_position: pd.Series
) -> pd.Series:
    """
    Apply a symmetric EWMAC trend filter to a single instrument.

    Parameters
    ----------
    adjusted_price : pd.Series
        Price series of the instrument.
    average_position : pd.Series
        Average position series (in contracts or normalized units).

    Returns
    -------
    pd.Series
        Position Series with sign inverted when EWMAC < 0.
    """
    filtered_position = copy(average_position)
    ewmac_values = ewmac(adjusted_price)
    bearish = ewmac_values < 0
    filtered_position[bearish] = -filtered_position[bearish]
    return filtered_position


def long_only_returns(
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    average_position_contracts_dict: dict,
    fx_series_dict: dict,
    cost_per_contract_dict: dict,
    multipliers: dict,
    capital: float,
) -> pd.Series:
    """
    Compute aggregated portfolio returns for a long-only strategy.

    Parameters
    ----------
    adjusted_prices_dict : dict
        Dictionary with adjusted price Series per instrument.
    std_dev_dict : dict
        Dictionary with annualized standard deviation objects or Series.
    average_position_contracts_dict : dict
        Dictionary with average positions in contracts per instrument.
    fx_series_dict : dict
        Dictionary with FX conversion Series for each instrument.
    cost_per_contract_dict : dict
        Dictionary with transaction costs per contract.
    multipliers : dict
        Contract multipliers for each instrument.
    capital : float
        Total portfolio capital for normalization.

    Returns
    -------
    pd.Series
        Aggregated percentage return Series across instruments.
    """
    perc_return_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=average_position_contracts_dict,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )
    perc_return_agg = aggregate_returns(perc_return_dict)
    return perc_return_agg


def calculate_position_dict_with_trend_forecast_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    fast_span: int = 64,
) -> dict:
    """
    Apply EWMAC-based trend forecast scaling to all instruments.

    Parameters
    ----------
    adjusted_prices_dict : dict
        Dictionary with adjusted price Series.
    average_position_contracts_dict : dict
        Dictionary with average positions in contracts.
    std_dev_dict : dict
        Dictionary with volatility estimation objects or Series.
    fast_span : int, optional
        Span for fast EWMA (default is 64).

    Returns
    -------
    dict
        Dictionary with position Series scaled by EWMAC forecast.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = {
        instrument_code: calculate_position_with_trend_forecast_applied(
            adjusted_prices_dict[instrument_code],
            average_position_contracts_dict[instrument_code],
            stdev_ann_perc=std_dev_dict[instrument_code],
            fast_span=fast_span,
        )
        for instrument_code in list_of_instruments
    }
    return position_dict_with_trend_filter


def calculate_position_with_trend_forecast_applied(
    adjusted_price: pd.Series,
    average_position: pd.Series,
    stdev_ann_perc,
    fast_span: int = 64,
) -> pd.Series:
    """
    Apply EWMAC forecast scaling to position for a single instrument.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series.
    average_position : pd.Series
        Average position (baseline position).
    stdev_ann_perc : standardDeviation
        Object or Series with volatility metrics.
    fast_span : int, optional
        Fast EWMA span (default is 64).

    Returns
    -------
    pd.Series
        Forecast-adjusted position Series (scaled by EWMAC signal strength).
    """
    forecast = calculate_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )
    return forecast * average_position / 10


def calculate_forecast_for_ewmac(
    adjusted_price: pd.Series, stdev_ann_perc, fast_span: int = 64
) -> pd.Series:
    """
    Compute a capped EWMAC forecast for a single instrument.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series.
    stdev_ann_perc : standardDeviation
        Volatility estimation object or Series.
    fast_span : int, optional
        Fast EWMA span (default is 64).

    Returns
    -------
    pd.Series
        EWMAC forecast capped between -20 and +20.
    """
    scaled_ewmac = calculate_scaled_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )
    capped_ewmac = scaled_ewmac.clip(-20, 20)
    return capped_ewmac


def calculate_scaled_forecast_for_ewmac(
    adjusted_price: pd.Series,
    stdev_ann_perc,
    fast_span: int = 64,
) -> pd.Series:
    """
    Compute scaled EWMAC forecast using empirically calibrated scaling constants.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series.
    stdev_ann_perc : standardDeviation
        Volatility estimation object or Series.
    fast_span : int, optional
        Fast EWMA span (default is 64).

    Returns
    -------
    pd.Series
        Scaled EWMAC forecast Series.
    """
    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    risk_adjusted_ewmac = calculate_risk_adjusted_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )
    forecast_scalar = scalar_dict[fast_span]
    scaled_ewmac = risk_adjusted_ewmac * forecast_scalar
    return scaled_ewmac


def calculate_risk_adjusted_forecast_for_ewmac(
    adjusted_price: pd.Series,
    stdev_ann_perc,
    fast_span: int = 64,
) -> pd.Series:
    """
    Compute risk-adjusted EWMAC forecast (normalized by daily price volatility).

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price series.
    stdev_ann_perc : standardDeviation
        Object or Series providing daily volatility via .daily_risk_price_terms().
    fast_span : int, optional
        Fast EWMA span (default is 64).

    Returns
    -------
    pd.Series
        Risk-adjusted EWMAC Series (EWMAC divided by daily volatility).
    """
    ewmac_values = ewmac(adjusted_price, fast_span=fast_span, slow_span=fast_span * 4)
    daily_price_vol = stdev_ann_perc.daily_risk_price_terms()
    risk_adjusted_ewmac = ewmac_values / daily_price_vol
    return risk_adjusted_ewmac


# ======================
#%% APPLY BUFFER POSITIONS
def apply_buffering_to_position_dict(
    position_contracts_dict: dict, average_position_contracts_dict: dict
) -> dict:
    """
    Apply position buffering instrument-by-instrument for a dictionary of Series.

    For each instrument code present in `position_contracts_dict`, this function
    calls `apply_buffering_to_positions` using the corresponding optimal position
    series and the average position series from `average_position_contracts_dict`.

    Parameters
    ----------
    position_contracts_dict : dict[str, pd.Series]
        Optimal position in contracts for each instrument (time series).
        Keys are instrument codes; values are pandas Series indexed by date/time.
    average_position_contracts_dict : dict[str, pd.Series]
        Average position in contracts for each instrument (time series), used
        to size the buffer around the optimal position.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary with the same keys. Each value is a pandas Series with the
        buffered position over time, indexed like the corresponding optimal
        position series.
    """
    instrument_list = list(position_contracts_dict.keys())
    buffered_position_dict = dict(
        [
            (
                instrument_code,
                apply_buffering_to_positions(
                    position_contracts=position_contracts_dict[instrument_code],
                    average_position_contracts=average_position_contracts_dict[
                        instrument_code
                    ],
                ),
            )
            for instrument_code in instrument_list
        ]
    )

    return buffered_position_dict


def apply_buffering_to_positions(
    position_contracts: pd.Series,
    average_position_contracts: pd.Series,
    buffer_size: float = 0.05,
) -> pd.Series:
    """
    Construct per-period upper/lower buffers around an optimal position path
    and apply the buffering rule.

    The buffer at each time t is computed as:
        buffer_t = abs(average_position_contracts_t) * buffer_size
    Upper and lower bands are then:
        upper_t = position_contracts_t + buffer_t
        lower_t = position_contracts_t - buffer_t
    The final buffered path is computed by `apply_buffer`.

    Parameters
    ----------
    position_contracts : pd.Series
        Target/optimal position in contracts over time.
    average_position_contracts : pd.Series
        Average position in contracts over time, used only to size the buffer.
    buffer_size : float, default 0.05
        Fraction of the absolute average position used as the buffer width.

    Returns
    -------
    pd.Series
        Buffered position series with the same index as `position_contracts`.
    """
    buffer = average_position_contracts.abs() * buffer_size
    upper_buffer = position_contracts + buffer
    lower_buffer = position_contracts - buffer

    buffered_position = apply_buffer(
        optimal_position=position_contracts,
        upper_buffer=upper_buffer,
        lower_buffer=lower_buffer,
    )

    return buffered_position


def apply_buffer(
    optimal_position: pd.Series, upper_buffer: pd.Series, lower_buffer: pd.Series
) -> pd.Series:
    """
    Apply a no-trade buffer to an optimal position time series.

    The rule is:
      - Start from the first observed optimal position (NaNs forward-filled).
        If the first value is NaN, start from 0.0.
      - For each subsequent period:
          * If last_position > upper_buffer_t, move to upper_buffer_t.
          * If last_position < lower_buffer_t, move to lower_buffer_t.
          * Otherwise, keep last_position unchanged.

    Inputs `upper_buffer` and `lower_buffer` are forward-filled and rounded
    before use. The output is not rounded.

    Parameters
    ----------
    optimal_position : pd.Series
        Optimal position path over time.
    upper_buffer : pd.Series
        Upper band for the buffer at each time.
    lower_buffer : pd.Series
        Lower band for the buffer at each time.

    Returns
    -------
    pd.Series
        Buffered position series aligned to `optimal_position.index`.
    """
    upper_buffer = upper_buffer.ffill().round()
    lower_buffer = lower_buffer.ffill().round()
    use_optimal_position = optimal_position.ffill()

    current_position = use_optimal_position[0]
    if np.isnan(current_position):
        current_position = 0.0

    buffered_position_list = [current_position]

    for idx in range(len(optimal_position.index))[1:]:
        current_position = apply_buffer_single_period(
            last_position=current_position,
            top_pos=upper_buffer[idx],
            bot_pos=lower_buffer[idx],
        )

        buffered_position_list.append(current_position)

    buffered_position = pd.Series(buffered_position_list, index=optimal_position.index)

    return buffered_position


def apply_buffer_single_period(last_position: int, top_pos: float, bot_pos: float):
    """
    One-step buffer update.

    If the last position is above the upper band, clamp to the upper band.
    If the last position is below the lower band, clamp to the lower band.
    Otherwise keep the last position unchanged.

    Parameters
    ----------
    last_position : int
        Position carried from the previous period.
    top_pos : float
        Upper buffer value for the current period.
    bot_pos : float
        Lower buffer value for the current period.

    Returns
    -------
    float
        Updated position for the current period according to the buffer rule.
    """
    if last_position > top_pos:
        return top_pos
    elif last_position < bot_pos:
        return bot_pos
    else:
        return last_position


def calculate_position_dict_with_multiple_trend_forecast_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    fast_spans: list,
) -> dict:
    """
    Apply multiple EWMAC forecasts to a dictionary of instruments.

    For each instrument in `adjusted_prices_dict`, compute a combined EWMAC
    forecast across multiple fast spans, and use it to scale the average
    position proportionally.

    Parameters
    ----------
    adjusted_prices_dict : dict[str, pd.Series]
        Adjusted price time series for each instrument.
    average_position_contracts_dict : dict[str, pd.Series]
        Average position in contracts for each instrument.
    std_dev_dict : dict[str, standardDeviation]
        Annualized volatility objects (or Series) used for forecast normalization.
    fast_spans : list[int]
        List of EWMAC fast spans to use when computing multiple forecasts.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary mapping each instrument code to its position series with
        multiple-trend forecast scaling applied.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = dict(
        [
            (
                instrument_code,
                calculate_position_with_multiple_trend_forecast_applied(
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


def calculate_position_with_multiple_trend_forecast_applied(
    adjusted_price: pd.Series,
    average_position: pd.Series,
    stdev_ann_perc,
    fast_spans: list,
) -> pd.Series:
    """
    Compute position scaling for a single instrument using multiple EWMAC forecasts.

    Combines several EWMAC signals (one for each `fast_span`), averages them,
    applies a forecast diversification multiplier (FDM), caps the forecast,
    and then scales the average position by forecast/10 as in Carver’s methodology.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price time series of the instrument.
    average_position : pd.Series
        Average position in contracts (baseline position).
    stdev_ann_perc : standardDeviation
        Annualized percentage standard deviation for volatility adjustment.
    fast_spans : list[int]
        List of fast spans used in the EWMAC forecasts.

    Returns
    -------
    pd.Series
        Position time series scaled by the combined multiple-trend forecast.
    """
    forecast = calculate_combined_ewmac_forecast(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_spans=fast_spans,
    )

    return forecast * average_position / 10


def calculate_combined_ewmac_forecast(
    adjusted_price: pd.Series,
    stdev_ann_perc,
    fast_spans: list,
) -> pd.Series:
    """
    Compute a combined EWMAC forecast across multiple fast spans.

    Each EWMAC forecast measures trend strength normalized by volatility.
    The forecasts are averaged equally, then scaled by a Forecast
    Diversification Multiplier (FDM) based on the number of independent
    trend rules, and finally capped between -20 and +20.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted price time series for the instrument.
    stdev_ann_perc : standardDeviation
        Annualized volatility measure used to scale forecasts.
    fast_spans : list[int]
        List of fast spans for individual EWMAC forecasts (e.g., [8, 16, 32]).

    Returns
    -------
    pd.Series
        Combined and volatility-scaled forecast series, clipped within [-20, 20].
    """
    all_forecasts_as_list = [
        calculate_forecast_for_ewmac(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            fast_span=fast_span,
        )
        for fast_span in fast_spans
    ]

    # Equally-weighted combination of all EWMAC forecasts
    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    # Forecast Diversification Multiplier (FDM)
    rule_count = len(fast_spans)
    FDM_DICT = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}
    fdm = FDM_DICT.get(rule_count, 1.26)  # default to max if >6

    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast
