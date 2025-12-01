import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm
from copy import copy


from _utils.core_functions import *

from _utils.normalized_price import calculate_normalised_price_dict
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

def relative_carry(
    instrument_code: str,
    adjusted_prices_dict: dict,  ## not used
    std_dev_dict: dict,
    carry_prices_dict: dict,
    asset_class_groupings:dict,
    span: int = 90,
) -> pd.Series:
    """
    Compute relative carry = instrument_carry - median_carry_of_asset_class.

    Parameters
    ----------
    instrument_code : str
        Instrument identifier.
    adjusted_prices_dict : dict
        Unused placeholder (included for API consistency).
    std_dev_dict : dict
        Dict mapping instrument → volatility measure.
    carry_prices_dict : dict
        Dict mapping instrument → carry series.
    span : int, default 90
        Span used in the carry forecast calculation.

    Returns
    -------
    pd.Series
        Relative carry series with zeros replaced by NaN.
    """
    carry_forecast = calculate_forecast_for_carry(
        stdev_ann_perc=std_dev_dict[instrument_code],
        carry_price=carry_prices_dict[instrument_code],
        span=span,
    )

    median_forecast = median_carry_for_instrument_in_asset_class(
        instrument_code=instrument_code,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
        asset_class_groupings=asset_class_groupings,
        span=span,
    )
    median_forecast_indexed = median_forecast.reindex(carry_forecast.index).ffill()

    relative_carry_forecast = carry_forecast - median_forecast_indexed
    relative_carry_forecast[relative_carry_forecast == 0] = np.nan

    return relative_carry_forecast


def median_carry_for_instrument_in_asset_class(
    instrument_code: str,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    asset_class_groupings:dict,
    span: int = 90
) -> pd.Series:
    """
    Compute the median carry forecast for the asset class of a given instrument.

    Parameters
    ----------
    instrument_code : str
        The instrument of interest.
    std_dev_dict : dict
        Volatility information for all instruments.
    carry_prices_dict : dict
        Carry series for all instruments.
    span : int
        Span parameter for carry forecast.

    Returns
    -------
    pd.Series
        Time-aligned median carry forecast for the asset class.
    """
    asset_class = get_asset_class_for_instrument(
        instrument_code,
        asset_class_groupings=asset_class_groupings,
    )

    median_carry = median_carry_for_asset_class(
        asset_class,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
        asset_class_groupings=asset_class_groupings,
        span=span,
    )

    return median_carry


def median_carry_for_asset_class(
    asset_class: str,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    asset_class_groupings: dict,
    span: int = 90,
) -> pd.Series:
    """
    Compute the median carry forecast across all instruments in an asset class.

    Parameters
    ----------
    asset_class : str
        Name of the asset class.
    std_dev_dict : dict
        Dict mapping instrument → volatility estimator.
    carry_prices_dict : dict
        Dict mapping instrument → carry series.
    asset_class_groupings : dict
        Mapping {asset_class: [instrument_codes]}.
    span : int
        Span parameter for carry forecast.

    Returns
    -------
    pd.Series
        Median carry forecast series across the asset class.
    """
    list_of_instruments = asset_class_groupings[asset_class]

    all_carry_forecasts = [
        calculate_forecast_for_carry(
            stdev_ann_perc=std_dev_dict[instrument_code],
            carry_price=carry_prices_dict[instrument_code],
            span=span,
        )
        for instrument_code in list_of_instruments
    ]

    all_carry_forecasts_pd = pd.concat(all_carry_forecasts, axis=1)
    median_carry = all_carry_forecasts_pd.median(axis=1)

    return median_carry

def get_asset_class_for_instrument(
    instrument_code: str,
    asset_class_groupings: dict
) -> str:
    """
    Determine the asset class (string) to which an instrument belongs.

    Procura o instrumento dentro das listas de cada asset class. 
    Assume-se que cada instrumento pertence a exatamente uma classe.

    Parameters
    ----------
    instrument_code : str
        Código do instrumento.
    asset_class_groupings : dict[str, list[str]]
        Dicionário com asset classes como chaves e instrumentos como valores.

    Returns
    -------
    str
        Nome da classe de ativo à qual o instrumento pertence.

    Raises
    ------
    ValueError
        Se o instrumento não pertencer a nenhuma classe.
    """
    possible_asset_classes = list(asset_class_groupings.keys())

    matches = [
        asset_class
        for asset_class in possible_asset_classes
        if instrument_code in asset_class_groupings[asset_class]
    ]

    if len(matches) == 0:
        raise ValueError(f"Instrument '{instrument_code}' not found in any asset class.")
    if len(matches) > 1:
        raise ValueError(f"Instrument '{instrument_code}' found in multiple asset classes: {matches}")

    return matches[0]