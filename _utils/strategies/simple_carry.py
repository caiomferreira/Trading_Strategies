import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm
from copy import copy
from scipy.interpolate import interp1d


from _utils.core_functions import *
from _utils.portfoliohandcrafiting import *
from _utils.strategies.trend_simple_filter import *
from _utils.seasonal_carry import calculate_seasonally_adjusted_carry


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





def calculate_position_dict_with_multiple_carry_forecast_applied(
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    average_position_contracts_dict: dict,
    carry_prices_dict: dict,
    carry_spans: list,
) -> dict:
    """
    Apply multiple carry-based forecasts across a portfolio of instruments.

    For each instrument in the input dictionaries, compute a combined carry forecast
    using several exponential smoothing spans. The combined forecast is scaled
    and applied to the average position to generate a risk-adjusted carry position.

    Parameters
    ----------
    adjusted_prices_dict : dict[str, pd.Series]
        Adjusted price time series for each instrument.
    std_dev_dict : dict[str, standardDeviation]
        Annualized volatility object for each instrument (used for risk normalization).
    average_position_contracts_dict : dict[str, pd.Series]
        Baseline position in contracts for each instrument.
    carry_prices_dict : dict[str, pd.DataFrame]
        DataFrame containing PRICE, CARRY, PRICE_CONTRACT, and CARRY_CONTRACT columns.
    carry_spans : list[int]
        List of smoothing spans for carry signals (e.g., [8, 16, 32]).

    Returns
    -------
    dict[str, pd.Series]
        Dictionary with carry-based position series for each instrument.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_carry = dict(
        [
            (
                instrument_code,
                calculate_position_with_multiple_carry_forecast_applied(
                    average_position=average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    carry_price=carry_prices_dict[instrument_code],
                    carry_spans=carry_spans,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_carry


def calculate_position_with_multiple_carry_forecast_applied(
    average_position: pd.Series,
    stdev_ann_perc,
    carry_price: pd.DataFrame,
    carry_spans: list,
) -> pd.Series:
    """
    Compute position scaling for a single instrument based on multiple carry forecasts.

    The carry forecast measures the relative value between near and far contracts,
    adjusted by volatility and smoothed across multiple horizons. The combined forecast
    is scaled by the average position and divided by 10 (Carver normalization).

    Parameters
    ----------
    average_position : pd.Series
        Average baseline position series.
    stdev_ann_perc : standardDeviation
        Annualized volatility measure used for normalization.
    carry_price : pd.DataFrame
        DataFrame with PRICE, CARRY, PRICE_CONTRACT, and CARRY_CONTRACT columns.
    carry_spans : list[int]
        List of smoothing spans for exponential weighting.

    Returns
    -------
    pd.Series
        Carry-based position series scaled by multiple forecasts.
    """
    forecast = calculate_combined_carry_forecast(
        stdev_ann_perc=stdev_ann_perc,
        carry_price=carry_price,
        carry_spans=carry_spans,
    )

    return forecast * average_position / 10


def calculate_combined_carry_forecast(
    stdev_ann_perc,
    carry_price: pd.DataFrame,
    carry_spans: list,
) -> pd.Series:
    """
    Combine multiple carry forecasts into a single normalized signal.

    Each carry forecast measures the forward premium/discount normalized by volatility,
    smoothed over different spans. Forecasts are equally weighted, scaled by a Forecast
    Diversification Multiplier (FDM), and capped within [-20, 20].

    Parameters
    ----------
    stdev_ann_perc : standardDeviation
        Annualized volatility measure for normalization.
    carry_price : pd.DataFrame
        DataFrame containing PRICE, CARRY, and contract codes.
    carry_spans : list[int]
        List of smoothing spans for carry signal calculation.

    Returns
    -------
    pd.Series
        Combined, scaled, and capped carry forecast series.
    """
    all_forecasts_as_list = [
        calculate_forecast_for_carry(
            stdev_ann_perc=stdev_ann_perc,
            carry_price=carry_price,
            span=span,
        )
        for span in carry_spans
    ]

    # Equal weighting of forecasts across spans
    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    # Forecast Diversification Multiplier (FDM)
    rule_count = len(carry_spans)
    FDM_DICT = {1: 1.0, 2: 1.02, 3: 1.03, 4: 1.04}
    fdm = FDM_DICT.get(rule_count, 1.04)

    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast


def calculate_forecast_for_carry(
    stdev_ann_perc,
    carry_price: pd.DataFrame,
    span: int,
) -> pd.Series:
    """
    Compute a single carry forecast smoothed by an exponential moving average.

    The carry is risk-adjusted by dividing by annualized price volatility, then
    exponentially smoothed and scaled by a constant factor (30). The final signal
    is clipped within [-20, 20].

    Parameters
    ----------
    stdev_ann_perc : standardDeviation
        Annualized volatility object for scaling the carry.
    carry_price : pd.DataFrame
        DataFrame containing PRICE, CARRY, PRICE_CONTRACT, and CARRY_CONTRACT columns.
    span : int
        Exponential smoothing span applied to the carry.

    Returns
    -------
    pd.Series
        Scaled and capped carry forecast series.
    """
    smooth_carry = calculate_smoothed_carry(
        stdev_ann_perc=stdev_ann_perc, carry_price=carry_price, span=span
    )
    scaled_carry = smooth_carry * 30
    capped_carry = scaled_carry.clip(-20, 20)

    return capped_carry


def calculate_smoothed_carry(
    stdev_ann_perc,
    carry_price: pd.DataFrame,
    span: int,
    use_seasonal_carry: bool = False,
    rolls_per_year: int = None,
) -> pd.Series:
    """
    Smooth the carry (either raw+vol-adjusted or seasonally-adjusted)
    using an exponential moving average.
    """

    # ==========================
    # 1) Carry normal (vol-adjusted)
    # ==========================
    risk_adj_carry = calculate_vol_adjusted_carry(
        stdev_ann_perc=stdev_ann_perc,
        carry_price=carry_price
    )

    # ==========================
    # 2) Seasonal adjustment opcional
    # ==========================
    if use_seasonal_carry:

        if rolls_per_year is None:
            raise ValueError("rolls_per_year must be provided when use_seasonal_carry=True")

        # Extrair série original de carry bruto
        # (Carver assume 'carry' é a primeira coluna do df)

        seasonally_adj_carry = calculate_seasonally_adjusted_carry(
            original_raw_carry=risk_adj_carry,
            rolls_per_year=rolls_per_year
        )

        # **Substitui o carry vol-adjusted pelo seasonal-adjusted**
        risk_adj_carry = seasonally_adj_carry

    # ==========================
    # 3) smoothing final
    # ==========================
    smooth_carry = risk_adj_carry.ewm(span).mean()

    return smooth_carry


def calculate_vol_adjusted_carry(
    stdev_ann_perc,
    carry_price: pd.DataFrame,
) -> pd.Series:
    """
    Compute the volatility-adjusted annualized carry.

    The raw carry is divided by annualized price volatility to produce a
    dimensionless signal comparable across assets.

    Parameters
    ----------
    stdev_ann_perc : standardDeviation
        Annualized volatility measure (used for normalization).
    carry_price : pd.DataFrame
        DataFrame containing PRICE, CARRY, and contract metadata.

    Returns
    -------
    pd.Series
        Volatility-adjusted annualized carry series.
    """
    ann_carry = calculate_annualised_carry(carry_price)
    ann_price_vol = stdev_ann_perc.annual_risk_price_terms()

    risk_adj_carry = ann_carry.ffill() / ann_price_vol.ffill()
    return risk_adj_carry


def calculate_annualised_carry(carry_price: pd.DataFrame) -> pd.Series:
    """
    Compute annualized carry between front and back contracts.

    The annualized carry is derived from the price difference divided by
    the time (in years) separating the two contract maturities.

    Parameters
    ----------
    carry_price : pd.DataFrame
        Must contain:
        - PRICE: front contract price
        - CARRY: back contract price
        - PRICE_CONTRACT, CARRY_CONTRACT: contract codes in YYYYMM format

    Returns
    -------
    pd.Series
        Annualized carry (price differential normalized by time to maturity).
    """
    raw_carry = carry_price.PRICE - carry_price.CARRY
    contract_diff = _total_year_frac_from_contract_series(
        carry_price.CARRY_CONTRACT
    ) - _total_year_frac_from_contract_series(carry_price.PRICE_CONTRACT)

    ann_carry = raw_carry / contract_diff
    return ann_carry


def _total_year_frac_from_contract_series(x: pd.Series) -> pd.Series:
    """
    Compute the total year fraction from a contract code series (YYYYMM).

    Parameters
    ----------
    x : pd.Series
        Series of contract identifiers as integers (e.g., 202503).

    Returns
    -------
    pd.Series
        Total year fraction (e.g., 2025.25 for March 2025).
    """
    years = _year_from_contract_series(x)
    month_frac = _month_as_year_frac_from_contract_series(x)
    return years + month_frac


def _year_from_contract_series(x: pd.Series) -> pd.Series:
    """Extract the year component (YYYY) from a contract code in YYYYMM format."""
    return x.floordiv(10000)


def _month_as_year_frac_from_contract_series(x: pd.Series) -> pd.Series:
    """Convert the month component (MM) from a YYYYMM contract into a fractional year."""
    return _month_from_contract_series(x) / 12.0


def _month_from_contract_series(x: pd.Series) -> pd.Series:
    """Extract the month component (MM) from a YYYYMM contract code."""
    return x.mod(10000) / 100.0


def calculate_position_dict_with_forecast_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    rule_spec: list,
) -> dict:
    """
    Generate a dictionary of final position time series after applying
    combined forecasts to each instrument.

    Parameters
    ----------
    adjusted_prices_dict : dict[str, pd.Series]
        Adjusted price series for each instrument.
    average_position_contracts_dict : dict[str, pd.Series]
        Baseline positions in contracts for each instrument.
    std_dev_dict : dict[str, standardDeviation]
        Annualized volatility estimators per instrument.
    carry_prices_dict : dict[str, pd.DataFrame]
        Carry-related price DataFrames per instrument.
    rule_spec : list[dict]
        List of rule specifications to compute forecasts (e.g. EWMAC, carry).

    Returns
    -------
    dict[str, pd.Series]
        Mapping from instrument code to final positions with forecasts applied.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_carry = dict(
        [
            (
                instrument_code,
                calculate_position_with_forecast_applied(
                    average_position=average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    carry_price=carry_prices_dict[instrument_code],
                    adjusted_price=adjusted_prices_dict[instrument_code],
                    rule_spec=rule_spec,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_carry

def calculate_position_with_forecast_applied(
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule_spec: list,
) -> pd.Series:
    """
    Compute the final position series for a single instrument after combining
    multiple forecast rules and applying the forecast to the baseline position.

    Parameters
    ----------
    average_position : pd.Series
        Baseline position in contracts.
    stdev_ann_perc : standardDeviation
        Annualized volatility estimator.
    carry_price : pd.DataFrame
        Carry-related price information for the instrument.
    adjusted_price : pd.Series
        Adjusted price series for the instrument.
    rule_spec : list[dict]
        Rule definitions to create the combined forecast.

    Returns
    -------
    pd.Series
        Final position series scaled by forecast/10.
    """
    forecast = calculate_combined_forecast(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        carry_price=carry_price,
        rule_spec=rule_spec,
    )

    return forecast * average_position / 10

def calculate_combined_forecast(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule_spec: list,
) -> pd.Series:
    """
    Combine multiple forecasting rules into a single forecast by:
    1. Computing each rule independently.
    2. Averaging the resulting forecasts equally.
    3. Scaling using the Forecast Diversification Multiplier (FDM).
    4. Capping the final forecast to [-20, 20].

    Parameters
    ----------
    stdev_ann_perc : standardDeviation
        Annualized volatility estimator.
    carry_price : pd.DataFrame
        Carry-related price information.
    adjusted_price : pd.Series
        Adjusted price series.
    rule_spec : list[dict]
        List of rule specifications.

    Returns
    -------
    pd.Series
        Combined, scaled and capped forecast series.
    """
    all_forecasts_as_list = [
        calculate_forecast(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            carry_price=carry_price,
            rule=rule,
        )
        for rule in rule_spec
    ]

    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    rule_count = len(rule_spec)
    fdm = get_fdm(rule_count)
    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast

def get_fdm(rule_count):
    """
    Retrieve the Forecast Diversification Multiplier (FDM) for a given
    number of forecasting rules via linear interpolation over known values.

    Parameters
    ----------
    rule_count : int
        Number of forecasting rules combined.

    Returns
    -------
    float
        Interpolated FDM value.
    """
    fdm = float(f_interp(rule_count))
    return fdm

def calculate_forecast(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule: dict,
) -> pd.Series:
    """
    Apply a single forecast rule (carry or EWMAC) using its parameters.

    Parameters
    ----------
    stdev_ann_perc : standardDeviation
        Annualized volatility estimator.
    carry_price : pd.DataFrame
        Carry-related price information.
    adjusted_price : pd.Series
        Adjusted price series.
    rule : dict
        Rule definition. Must include:
        - "function": str → "carry" or "ewmac"
        - parameters consistent with the chosen function.

    Returns
    -------
    pd.Series
        Forecast series computed for the selected rule.

    Raises
    ------
    Exception
        If the rule function is not recognized.
    """
    if rule["function"] == "carry":
        span = rule["span"]
        forecast = calculate_forecast_for_carry(
            stdev_ann_perc=stdev_ann_perc, carry_price=carry_price, span=span
        )

    elif rule["function"] == "ewmac":
        fast_span = rule["fast_span"]
        forecast = calculate_forecast_for_ewmac(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            fast_span=fast_span,
        )

    else:
        raise Exception("Rule %s not recognised!" % rule["function"])

    return forecast
