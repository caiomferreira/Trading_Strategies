import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm
from copy import copy
from scipy.interpolate import interp1d


from _utils.core_functions import *
from _utils.portfoliohandcrafiting import *

from _utils.strategies.simple_carry import calculate_annualised_carry

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

SECONDS_IN_YEAR = 60 * 60 * 24 * 365.25


def calculate_synthetic_spot_dict(
    adjusted_prices_dict: dict, carry_prices_dict: dict
) -> dict:
    """
    Compute synthetic spot price series for multiple instruments.

    Parameters
    ----------
    adjusted_prices_dict : dict
        Mapping {instrument_code: adjusted_price_series}.
    carry_prices_dict : dict
        Mapping {instrument_code: carry_price_series}.

    Returns
    -------
    dict
        Mapping {instrument_code: synthetic_spot_series}.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())
    synthetic_spot_dict = dict(
        [
            (
                instrument_code,
                calculate_synthetic_spot(
                    adjusted_prices_dict[instrument_code],
                    carry_price=carry_prices_dict[instrument_code],
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return synthetic_spot_dict


def calculate_synthetic_spot(
    adjusted_price: pd.Series, carry_price: pd.Series
) -> pd.Series:
    """
    Construct the synthetic spot price by removing accumulated carry
    from the adjusted futures price.

    This creates a synthetic spot-like series consistent with the
    implied carry curve.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted futures price series.
    carry_price : pd.Series
        Carry price series (used to derive annualised carry).

    Returns
    -------
    pd.Series
        Synthetic spot series.
    """
    ann_carry = calculate_annualised_carry(carry_price)
    diff_index_in_years_as_pd = pd_series_of_diff_index_in_years(ann_carry)

    carry_per_period = diff_index_in_years_as_pd * ann_carry
    cum_carry = carry_per_period.cumsum()
    syn_spot = adjusted_price - cum_carry

    return syn_spot


def pd_series_of_diff_index_in_years(x: pd.Series):
    """
    Convert index time differences into year fractions and return them as a Series.

    The first element is forced to zero to align with the original time series length.

    Parameters
    ----------
    x : pd.Series
        Input time series with a DatetimeIndex.

    Returns
    -------
    pd.Series
        Series of year-fraction intervals aligned with x.index.
    """
    diff_index_in_years = get_annual_intervals_from_series(x)

    return pd.Series([0] + diff_index_in_years, x.index)


def get_annual_intervals_from_series(x: pd.Series):
    """
    Compute the time difference between consecutive index timestamps
    expressed in years.

    Parameters
    ----------
    x : pd.Series
        Input series with DatetimeIndex.

    Returns
    -------
    list of float
        List containing the fractional-year difference between each timestamp.
    """
    data = x.copy()
    data.index = pd.to_datetime(data.index)
    diff_index = data[1:].index - data[:-1].index
    diff_index_as_list = list(diff_index)
    diff_index_in_seconds = [
        index_item.total_seconds() for index_item in diff_index_as_list
    ]
    diff_index_in_years = [
        index_item_in_seconds / SECONDS_IN_YEAR
        for index_item_in_seconds in diff_index_in_seconds
    ]

    return diff_index_in_years
