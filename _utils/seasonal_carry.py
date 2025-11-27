import pandas as pd
import numpy as np
import datetime
from copy import copy


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


# =====================================================================
# Global constants
# =====================================================================

CALENDAR_DAYS_IN_YEAR = 365.25
NOTIONAL_YEAR = 2001                 # Must not be a leap year
NEXT_NOTIONAL_YEAR = NOTIONAL_YEAR + 1
SECONDS_IN_YEAR = 60 * 60 * 24 * 365.25


# =====================================================================
# Seasonal Adjustment — Main Function
# =====================================================================
def calculate_seasonally_adjusted_carry(
    original_raw_carry: pd.Series, rolls_per_year: int
) -> pd.Series:
    """
    Compute seasonally adjusted carry for a futures series.

    The method:
      - extracts average seasonal patterns,
      - shifts the seasonal component according to the roll frequency,
      - reindexes seasonal elements to the original dates,
      - removes the original seasonal effect and adds the shifted one.

    Parameters
    ----------
    original_raw_carry : pd.Series
        Unadjusted carry time series.
    rolls_per_year : int
        Number of contract rolls per year (e.g. 4 for quarterly).

    Returns
    -------
    pd.Series
        Seasonally adjusted carry series aligned with the original index.
    """
    original_index = original_raw_carry.index
    original_raw_carry.index = pd.to_datetime(original_raw_carry.index)
    average_seasonal = calculate_average_seasonal(original_raw_carry)
    shifted_avg_seasonal = calculate_shifted_avg_seasonal(
        average_seasonal=average_seasonal,
        rolls_per_year=rolls_per_year
    )

    # Forward-looking alignment warning is in original comments
    average_seasonal_indexed = reindex_seasonal_component_to_index(
        average_seasonal, original_raw_carry.index
    )
    shifted_avg_seasonal_indexed = reindex_seasonal_component_to_index(
        shifted_avg_seasonal, original_raw_carry.index
    )

    net_seasonally_adjusted_carry = original_raw_carry - average_seasonal_indexed
    correctly_seasonally_adjusted_carry = (
        net_seasonally_adjusted_carry + shifted_avg_seasonal_indexed
    )
    correctly_seasonally_adjusted_carry.index = original_index

    return correctly_seasonally_adjusted_carry


# =====================================================================
# Seasonal Extraction and Averaging
# =====================================================================

def calculate_average_seasonal(original_raw_carry: pd.Series) -> pd.Series:
    """
    Compute the average seasonal pattern of a carry series.

    Steps:
      - resample to daily frequency,
      - forward-fill,
      - compute 365-day rolling average,
      - extract seasonality as deviation from rolling mean,
      - convert to seasonal matrix,
      - compute exponentially weighted mean across years.

    Parameters
    ----------
    original_raw_carry : pd.Series
        Original carry series.

    Returns
    -------
    pd.Series
        Average daily seasonal pattern (notional year index).
    """
    original_raw_carry_calendar_days = original_raw_carry.resample("1D").mean()
    original_raw_carry_ffill = original_raw_carry_calendar_days.ffill()

    rolling_average = original_raw_carry_ffill.rolling(365).mean()
    seasonal_component = original_raw_carry_ffill - rolling_average

    seasonal_component_as_matrix = seasonal_matrix(
        seasonal_component, notional_year=NOTIONAL_YEAR
    )

    average_seasonal = (
        seasonal_component_as_matrix.transpose().ewm(5).mean().iloc[-1]
    )

    return average_seasonal


def calculate_shifted_avg_seasonal(
    average_seasonal: pd.Series, rolls_per_year: int
) -> pd.Series:
    """
    Shift the seasonal curve forward according to roll frequency.

    Parameters
    ----------
    average_seasonal : pd.Series
        Average seasonal pattern tied to the notional year.
    rolls_per_year : int
        Number of rolls per year.

    Returns
    -------
    pd.Series
        Shifted seasonal pattern (still in notional year index).
    """
    shift_days = int(CALENDAR_DAYS_IN_YEAR / rolls_per_year)

    shifted_avg_seasonal = shift_seasonal_series(
        average_seasonal, shift_days=shift_days
    )

    return shifted_avg_seasonal


# =====================================================================
# Seasonal Matrix Utilities
# =====================================================================

def seasonal_matrix(x, notional_year=NOTIONAL_YEAR):
    """
    Build a matrix where each column corresponds to one calendar year
    of the input series, mapped into a fixed notional year.

    Parameters
    ----------
    x : pd.Series
        Daily seasonal component series.
    notional_year : int
        Year to which all dates are mapped.

    Returns
    -------
    pd.DataFrame
        Seasonal matrix with index = notional year days,
        columns = actual years in the series.
    """
    years_to_use = unique_years_in_index(x.index)

    list_of_slices = [
        produce_list_from_x_for_year(x, year, notional_year=notional_year)
        for year in years_to_use
    ]

    concat_list = pd.concat(list_of_slices, axis=1)
    concat_list.columns = years_to_use

    concat_list = concat_list.sort_index()  # handles leap year cleanup

    return concat_list


def shift_seasonal_series(average_seasonal: pd.Series, shift_days: int):
    """
    Shift the seasonal curve forward by a certain number of days.

    Used so carry is aligned with the contract being rolled into.

    Parameters
    ----------
    average_seasonal : pd.Series
        Average seasonal pattern.
    shift_days : int
        Number of days to shift forward.

    Returns
    -------
    pd.Series
        Shifted seasonal pattern (notional-year indexed).
    """
    next_year = NEXT_NOTIONAL_YEAR
    next_year_seasonal = set_year_to_notional_year(
        average_seasonal, notional_year=next_year
    )

    two_years_worth = pd.concat([average_seasonal, next_year_seasonal], axis=0)
    shifted_two_years_worth = two_years_worth.shift(shift_days)

    shifted_average_seasonal_matrix = seasonal_matrix(shifted_two_years_worth)
    shifted_average_seasonal = (
        shifted_average_seasonal_matrix.transpose().ffill().iloc[-1].transpose()
    )

    return shifted_average_seasonal


def reindex_seasonal_component_to_index(seasonal_component, index):
    """
    Expand the seasonal component across all years present in the tar-
    get index, reassigning notional-year dates to those real years.

    Parameters
    ----------
    seasonal_component : pd.Series
        Seasonal curve defined over notional year.
    index : pd.DatetimeIndex
        Target index.

    Returns
    -------
    pd.Series
        Seasonal component aligned with the target index via ffill().
    """
    all_years = unique_years_in_index(index)

    data_with_years = [
        set_year_to_notional_year(seasonal_component, notional_year)
        for notional_year in all_years
    ]

    sequenced_data = pd.concat(data_with_years, axis=0)
    aligned_seasonal = sequenced_data.reindex(index, method="ffill")

    return aligned_seasonal


# =====================================================================
# Year and Index Utilities
# =====================================================================

def unique_years_in_index(index):
    """
    Extract sorted unique years from a DatetimeIndex.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index from which to extract years.

    Returns
    -------
    list of int
        Sorted list of unique years.
    """
    all_years = years_in_index(index)
    unique_years = list(set(all_years))
    unique_years.sort()
    return unique_years


def produce_list_from_x_for_year(x, year, notional_year=NOTIONAL_YEAR):
    """
    Produce a slice of the series for a specific calendar year and
    map its dates into the notional year.

    Parameters
    ----------
    x : pd.Series
        Input time series.
    year : int
        Year to extract.
    notional_year : int
        Target notional year.

    Returns
    -------
    pd.Series
        Values with the same calendar-day structure but mapped
        into the notional year.
    """
    list_of_matching_points = index_matches_year(x.index, year)
    matched_x = x[list_of_matching_points]

    matched_x_notional_year = set_year_to_notional_year(
        matched_x, notional_year=notional_year
    )

    return matched_x_notional_year


def set_year_to_notional_year(x, notional_year=NOTIONAL_YEAR):
    """
    Change the year of each index entry in a series to a fixed notional year.

    Parameters
    ----------
    x : pd.Series
        Input series with DatetimeIndex.
    notional_year : int
        Year to map all timestamps into.

    Returns
    -------
    pd.Series
        Series indexed with notional-year dates.
    """
    y = copy(x)
    new_index = [
        change_index_day_to_notional_year(index_item, notional_year)
        for index_item in list(x.index)
    ]
    y.index = new_index
    return y


def change_index_day_to_notional_year(index_item, notional_year=NOTIONAL_YEAR):
    """
    Replace the year of a date with the given notional year.

    Parameters
    ----------
    index_item : datetime.date or datetime.datetime
        Original date.
    notional_year : int
        Target year.

    Returns
    -------
    datetime.date
        Date with replaced year.
    """
    return datetime.date(
        notional_year, index_item.month, index_item.day
    )


def index_matches_year(index, year):
    """
    Boolean mask: whether each index date belongs to a year (excluding Feb 29).

    Parameters
    ----------
    index : pd.DatetimeIndex
        Input index.
    year : int
        Year to match.

    Returns
    -------
    list of bool
        Mask indicating dates belonging to the year (no leap-day contamination).
    """
    return [
        _index_matches_no_leap_days(index_value, year)
        for index_value in index
    ]


def _index_matches_no_leap_days(index_value, year_to_match):
    """
    Helper: check if a given date matches a year, excluding Feb 29.

    Parameters
    ----------
    index_value : datetime.date or datetime.datetime
    year_to_match : int

    Returns
    -------
    bool
    """
    if index_value.year != year_to_match:
        return False

    if index_value.month != 2:
        return True

    if index_value.day == 29:
        return False

    return True


def years_in_index(index):
    """
    Extract all years from a DatetimeIndex as a list.

    Parameters
    ----------
    index : pd.DatetimeIndex

    Returns
    -------
    list of int
    """
    index_list = list(index)
    all_years = [item.year for item in index_list]
    return all_years
