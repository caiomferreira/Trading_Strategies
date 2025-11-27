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
from _utils.strategies.trend_carry import *

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

def calculate_position_dict_with_forecast_and_vol_scalar_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    rule_spec: list,
    rolls_per_year: dict = None,
    use_seasonal_carry: bool = False,
    apply_vol_regime_to_carry: bool = True,
    apply_vol_regime_to_emac: bool = True,
) -> dict:
    """
    Build a dictionary of position series for multiple instruments using
    combined forecasts (EWMAC + Carry) with optional volatility-scaling
    and optional seasonal carry.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())

    position_dict_with_carry = {
        instrument_code: calculate_position_with_forecast_and_vol_scalar_applied(
            average_position=average_position_contracts_dict[instrument_code],
            stdev_ann_perc=std_dev_dict[instrument_code],
            carry_price=carry_prices_dict[instrument_code],
            adjusted_price=adjusted_prices_dict[instrument_code],
            rule_spec=rule_spec,
            use_seasonal_carry=use_seasonal_carry,
            rolls_per_year=(
                rolls_per_year.get(instrument_code)
                if isinstance(rolls_per_year, dict)
                else None
            ),
            apply_vol_regime_to_carry=apply_vol_regime_to_carry,
            apply_vol_regime_to_emac=apply_vol_regime_to_emac,
        )
        for instrument_code in list_of_instruments
    }

    return position_dict_with_carry




def calculate_position_with_forecast_and_vol_scalar_applied(
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule_spec: list,
    use_seasonal_carry: bool,
    rolls_per_year: int,
    apply_vol_regime_to_carry: bool = True,
    apply_vol_regime_to_emac: bool = True,
) -> pd.Series:
    """
    Compute position series for a single instrument using multiple forecast rules.
    """
    forecast = calculate_combined_forecast_with_vol_scalar_applied(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        carry_price=carry_price,
        rule_spec=rule_spec,
        use_seasonal_carry=use_seasonal_carry,
        rolls_per_year=rolls_per_year,
        apply_vol_regime_to_carry=apply_vol_regime_to_carry,
        apply_vol_regime_to_emac=apply_vol_regime_to_emac,
    )

    return forecast * average_position / 10


def calculate_combined_forecast_with_vol_scalar_applied(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule_spec: list,
    use_seasonal_carry: bool,
    rolls_per_year: int,
    apply_vol_regime_to_carry: bool = True,
    apply_vol_regime_to_emac: bool = True,
) -> pd.Series:
    """
    Combine forecasts from all rules (EWMAC and Carry).
    """
    all_forecasts_as_list = [
        calculate_forecast_with_vol_scalar_applied(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            carry_price=carry_price,
            rule=rule,
            use_seasonal_carry=use_seasonal_carry,
            rolls_per_year=rolls_per_year,
            apply_vol_regime_to_carry=apply_vol_regime_to_carry,
            apply_vol_regime_to_emac=apply_vol_regime_to_emac,
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


def calculate_forecast_with_vol_scalar_applied(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule: dict,
    use_seasonal_carry: bool,
    rolls_per_year: int,
    apply_vol_regime_to_carry: bool = True,
    apply_vol_regime_to_emac: bool = True,
) -> pd.Series:
    """
    Dispatch carry or EWMAC forecast.
    """
    if rule["function"] == "carry":
        span = rule["span"]
        forecast = calculate_forecast_for_carry_with_optional_vol_scaling(
            stdev_ann_perc=stdev_ann_perc,
            carry_price=carry_price,
            span=span,
            use_seasonal_carry=use_seasonal_carry ,
            rolls_per_year=rolls_per_year,
            apply_vol_regime_to_carry=apply_vol_regime_to_carry,
        )

    elif rule["function"] == "ewmac":
        fast_span = rule["fast_span"]
        forecast = calculate_forecast_for_ewmac_with_optional_vol_scaling(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            fast_span=fast_span,
            apply_vol_regime_to_emac=apply_vol_regime_to_emac,
        )

    else:
        raise Exception(f"Rule {rule['function']} not recognised!")

    return forecast

def calculate_forecast_for_carry_with_optional_vol_scaling(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    span: int,
    apply_vol_regime_to_carry: bool = True,
    use_seasonal_carry: bool = False,
    rolls_per_year: int = None,
):
    """
    Carry forecast with optional vol scaling and optional seasonal adjustment.
    """

    smooth_carry = calculate_smoothed_carry(
        stdev_ann_perc=stdev_ann_perc,
        carry_price=carry_price,
        span=span,
        use_seasonal_carry=use_seasonal_carry,
        rolls_per_year=rolls_per_year,
    )

    if apply_vol_regime_to_carry:
        smooth_carry = apply_vol_regime_to_forecast(
            smooth_carry, stdev_ann_perc=stdev_ann_perc
        )
        scaled_carry = smooth_carry * 23
    else:
        scaled_carry = smooth_carry * 30

    capped_carry = scaled_carry.clip(-20, 20)
    return capped_carry


def calculate_forecast_for_ewmac_with_optional_vol_scaling(
    adjusted_price: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_span: int = 64,
    apply_vol_regime_to_emac: bool = True,
):
    """
    EWMAC forecast with optional vol regime.
    """
    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    risk_adjusted_ewmac = calculate_risk_adjusted_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )

    if apply_vol_regime_to_emac:
        risk_adjusted_ewmac = apply_vol_regime_to_forecast(
            risk_adjusted_ewmac, stdev_ann_perc=stdev_ann_perc
        )

    forecast_scalar = scalar_dict[fast_span]
    scaled_ewmac = risk_adjusted_ewmac * forecast_scalar
    capped_ewmac = scaled_ewmac.clip(-20, 20)

    return capped_ewmac


def apply_vol_regime_to_forecast(
    scaled_forecast: pd.Series, stdev_ann_perc: pd.Series
) -> pd.Series:
    """
    Apply volatility-regime attenuation to a forecast.

    Parameters
    ----------
    scaled_forecast : pd.Series
        Base forecast before attenuation.
    stdev_ann_perc : pd.Series or standardDeviation
        Volatility information used to compute attenuation.

    Returns
    -------
    pd.Series
        Volatility-adjusted forecast.
    """
    smoothed_vol_attenuation = get_attenuation(scaled_forecast)
    return scaled_forecast * smoothed_vol_attenuation


def get_attenuation(stdev_ann_perc: standardDeviation) -> pd.Series:
    """
    Compute volatility-based attenuation factors using quantiles
    and an exponential smoothing step.

    Parameters
    ----------
    stdev_ann_perc : standardDeviation
        Volatility measurements.

    Returns
    -------
    pd.Series
        Smoothed volatility attenuation factors.
    """
    normalised_vol = calculate_normalised_vol(stdev_ann_perc)
    normalised_vol_q = quantile_of_points_in_data_series(normalised_vol)
    vol_attenuation = normalised_vol_q.apply(multiplier_function)
    smoothed_vol_attenuation = vol_attenuation.ewm(span=10).mean()

    return smoothed_vol_attenuation


def multiplier_function(vol_quantile):
    """
    Convert a volatility quantile into an attenuation multiplier.

    Parameters
    ----------
    vol_quantile : float
        Quantile value (0 to 1).

    Returns
    -------
    float
        Attenuation multiplier.
    """
    if np.isnan(vol_quantile):
        return 1.0

    return 2 - 1.5 * vol_quantile


def calculate_normalised_vol(stdev_ann_perc: standardDeviation) -> pd.Series:
    """
    Normalise volatility by dividing by its 10-year rolling average.

    Parameters
    ----------
    stdev_ann_perc : standardDeviation
        Annualised standard deviation series.

    Returns
    -------
    pd.Series
        Normalised volatility.
    """
    ten_year_averages = stdev_ann_perc.rolling(2500, min_periods=10).mean()
    return stdev_ann_perc / ten_year_averages


def quantile_of_points_in_data_series(data_series):
    """
    Compute a running quantile value for each point in a series,
    using only past data points at each step.

    Parameters
    ----------
    data_series : pd.Series
        Input series.

    Returns
    -------
    pd.Series
        Series of quantile values (0 to 1).
    """
    numpy_series = np.array(data_series)
    results = []

    for irow in range(len(data_series)):
        current_value = numpy_series[irow]
        count_less_than = (numpy_series < current_value)[:irow].sum()
        results.append(count_less_than / (irow + 1))

    results_series = pd.Series(results, index=data_series.index)
    return results_series


