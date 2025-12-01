import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm
from copy import copy
from scipy.interpolate import interp1d

from dataclasses import dataclass
import datetime


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

# --------------------------------------------------
# ENUMS AND CONSTANTS
# --------------------------------------------------
Frequency = Enum("Frequency", "Natural Year Month Week BDay")

DEFAULT_DATE_FORMAT = "%Y-%m-%d"

NATURAL = Frequency.Natural
YEAR = Frequency.Year
MONTH = Frequency.Month
WEEK = Frequency.Week
BDAY = Frequency.BDay

BUSINESS_DAYS_IN_YEAR = 252
WEEKS_PER_YEAR = 52.25
MONTHS_PER_YEAR = 12
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

PERIODS_PER_YEAR = {
    MONTH: MONTHS_PER_YEAR,
    WEEK: WEEKS_PER_YEAR,
    YEAR: 1,
}

QUANTILE_EXTREME = 0.01
QUANTILE_STD = 0.3
NORMAL_RATIO = norm.ppf(QUANTILE_EXTREME) / norm.ppf(QUANTILE_STD)


FORECAST_SCALAR = 9.3
AVG_ABS_FORECAST = 10.0


# SIGNAL CREATION

FDM_LIST = {
    1: 1.0,
    2: 1.02,
    3: 1.03,
    4: 1.23,
    5: 1.25,
    6: 1.27,
    7: 1.29,
    8: 1.32,
    9: 1.34,
    10: 1.35,
    11: 1.36,
    12: 1.38,
    13: 1.39,
    14: 1.41,
    15: 1.42,
    16: 1.44,
    17: 1.46,
    18: 1.48,
    19: 1.50,
    20: 1.53,
    21: 1.54,
    22: 1.55,
    25: 1.69,
    30: 1.81,
    35: 1.93,
    40: 2.00,
}
fdm_x = list(FDM_LIST.keys())
fdm_y = list(FDM_LIST.values())

f_interp = interp1d(fdm_x, fdm_y, bounds_error=False, fill_value=2)


# --------------------------------------------------
# FREQUENCY HANDLING
# --------------------------------------------------
PERIODS_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "yearly": 1}
SECONDS_PER_YEAR = 365 * 24 * 60 * 60


DATA_START = datetime.datetime(2000,1,1)
# DATA_START = '2000-01-01'

def periods_per_year(freq: Frequency):
    """
    Returns number of periods per year for the given frequency.
    """
    return PERIODS_YEAR.get(freq, BUSINESS_DAYS_IN_YEAR)


def years_in_data(series: pd.Series) -> float:
    """
    Calculates number of years covered by the dataset.
    """
    diff = series.index[-1] - series.index[0]
    return diff.total_seconds() / SECONDS_PER_YEAR


def sum_by_frequency(perc_return: pd.Series, freq: Frequency = NATURAL) -> pd.Series:
    """
    Aggregates returns according to the chosen frequency.
    """
    if freq == NATURAL:
        return perc_return

    freq_map = {YEAR: "Y", WEEK: "7D", MONTH: "ME"}
    freq_str = freq_map[freq]
    return perc_return.resample(freq_str).sum()


# --------------------------------------------------
# ANNUALIZED METRICS
# --------------------------------------------------
def annualized_mean(return_freq: pd.Series, freq: Frequency) -> float:
    """
    Annualized mean return based on given frequency.
    """
    return return_freq.mean() * periods_per_year(freq)


def annualized_std(return_freq: pd.Series, freq: Frequency) -> float:
    """
    Annualized standard deviation (volatility) based on given frequency.
    """
    return return_freq.std() * np.sqrt(periods_per_year(freq))


# --------------------------------------------------
# DRAWDOWN
# --------------------------------------------------
def calculate_drawdown(perc_return: pd.Series) -> pd.Series:
    """
    Calculates drawdowns as deviation from previous peak of cumulative return.
    """
    cum_return = perc_return.cumsum()
    max_cum = cum_return.rolling(len(perc_return) + 1, min_periods=1).max()
    return max_cum - cum_return


# --------------------------------------------------
# QUANTILE RATIOS
# --------------------------------------------------
def quantile_ratio_lower(x: pd.Series) -> float:
    """
    Lower tail quantile ratio vs normal distribution baseline.
    """
    x_clean = demean_remove_zeros(x)
    raw_ratio = x_clean.quantile(QUANTILE_EXTREME) / x_clean.quantile(QUANTILE_STD)
    return raw_ratio / NORMAL_RATIO


def quantile_ratio_upper(x: pd.Series) -> float:
    """
    Upper tail quantile ratio vs normal distribution baseline.
    """
    x_clean = demean_remove_zeros(x)
    raw_ratio = x_clean.quantile(1 - QUANTILE_EXTREME) / x_clean.quantile(1 - QUANTILE_STD)
    return raw_ratio / NORMAL_RATIO


def demean_remove_zeros(x: pd.Series) -> pd.Series:
    """
    Replaces zeros with NaN and de-means the series.
    """
    x[x == 0] = np.nan
    return x - x.mean()


# --------------------------------------------------
# AUXILIARY FUNCTIONS
# --------------------------------------------------

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

# --------------------------------------------------
# RISK AND CAPITAL CALCULATIONS
# --------------------------------------------------
def calculate_minimum_capital(multiplier: float,
                              price: float,
                              fx: float,
                              ann_volatility: float,
                              target_risk: float,
                              n_contracts: int = 5) -> float:
    """
    Minimum capital required to reach a target annualized risk.
    """
    min_capital = n_contracts * multiplier * price * fx * ann_volatility / target_risk
    return min_capital


def calculate_fixed_risk_position(capital: float,
                                  target_risk_tau: float,
                                  price: pd.Series,
                                  fx: pd.Series,
                                  multiplier: float,
                                  ann_volatility: float) -> pd.Series:
    """
    Calculates number of contracts to hold given a fixed target risk (τ).
    Formula: N = (Capital x τ) / (Multiplier x Price x FX x σ%)
    """
    position_contracts = capital * target_risk_tau / (multiplier * price * fx * ann_volatility)
    return position_contracts


def calculate_target_risk_std(adjusted_price: pd.Series, current_price: pd.Series) -> float:
    """
    Calculates annualized volatility (standard deviation) of daily percentage returns
    over the most recent 30 business days.

    Uses current_price-relative returns based on adjusted_price and reference current_price series.
    """
    daily_returns = adjusted_price.diff() / current_price.shift(1)
    std = daily_returns.tail(30).std()
    return std * np.sqrt(BUSINESS_DAYS_IN_YEAR)


def calculate_variable_target_risk_std(
    adjusted_price: pd.Series,
    current_price: pd.Series,
    use_perc_returns: bool = True,
    annualise_stdev: bool = True,  
            
) -> pd.Series:
    """
    Computes a dynamic (variable) annualized volatility estimate using an
    exponentially weighted moving standard deviation of daily returns.

    Parameters
    ----------
    adjusted_price : pd.Series
        Adjusted close current_prices.
    current_price : pd.Series
        Reference current_price series.
    use_perc_returns : bool, default=True
        If True, computes daily percentage returns. If False, uses absolute differences.
    annualise_stdev : bool, default=True
        If True, scales standard deviation to annualized volatility using √252.

    Returns
    -------
    pd.Series
        Weighted annualized volatility estimate combining short-term (EWM)
        and long-term (10-year) volatility with weights 0.7 and 0.3 respectively.
    """
    if use_perc_returns:
        daily_returns = adjusted_price.diff() / current_price.shift(1)
    else:
        daily_returns = adjusted_price.diff()

    # Exponentially weighted daily standard deviation
    exp_daily_std = daily_returns.ewm(span=32).std()

    if annualise_stdev:
        annual_factor = np.sqrt(BUSINESS_DAYS_IN_YEAR)
    else:
        annual_factor = 1  # keep daily volatility

    annualized_std = exp_daily_std * annual_factor

    # 10-year ou max data rolling volatility mean
    std_long_term = annualized_std.rolling(BUSINESS_DAYS_IN_YEAR * 10, min_periods=1).mean()

    # Weighted combination (30% long-term, 70% recent)
    weighted_std = 0.3 * std_long_term + 0.7 * annualized_std

    return weighted_std

class standardDeviation(pd.Series):
    def __init__(
        self,
        adjusted_price: pd.Series,
        current_price: pd.Series,
        use_perc_returns: bool = True,
        annualise_stdev: bool = True,
    ):

        stdev = calculate_variable_target_risk_std(
            adjusted_price=adjusted_price,
            current_price=current_price,
            annualise_stdev=annualise_stdev,
            use_perc_returns=use_perc_returns,
        )
        super().__init__(stdev)

        self._use_perc_returns = use_perc_returns
        self._annualised = annualise_stdev
        self._current_price = current_price

    def daily_risk_price_terms(self):
        stdev = copy(self)
        if self.annualised:
            stdev = stdev / (BUSINESS_DAYS_IN_YEAR ** 0.5)

        if self.use_perc_returns:
            stdev = stdev * self.current_price

        return stdev

    def annual_risk_price_terms(self):
        stdev = copy(self)
        if not self.annualised:
            # daily
            stdev = stdev * (BUSINESS_DAYS_IN_YEAR ** 0.5)

        if self.use_perc_returns:
            stdev = stdev * self.current_price

        return stdev

    @property
    def annualised(self) -> bool:
        return self._annualised

    @property
    def use_perc_returns(self) -> bool:
        return self._use_perc_returns

    @property
    def current_price(self) -> pd.Series:
        return self._current_price

def calculate_position_series_given_variable_risk(
    capital: float,
    risk_target_tau: float,
    fx: pd.Series,
    multiplier: float,
    instrument_risk: standardDeviation,
) -> pd.Series:

    # N = (Capital × τ) ÷ (Multiplier × Price × FX × σ %)
    ## resolves to N = (Capital × τ) ÷ (Multiplier × FX × daily stdev price terms × 16)
    ## for simplicity we use the daily risk in price terms, even if we calculated annualised % returns
    daily_risk_price_terms = instrument_risk.daily_risk_price_terms()
    
    weigth_position_vol = capital * risk_target_tau / (multiplier * fx * daily_risk_price_terms * (BUSINESS_DAYS_IN_YEAR ** 0.5))

    return weigth_position_vol

# --------------------------------------------------
# RETURN CALCULATION
# --------------------------------------------------
def calculate_perc_returns(position_contracts_held: pd.Series,
                           adjusted_price: pd.Series,
                           multiplier: float,
                           fx_series: pd.Series,
                           capital_required: pd.Series) -> pd.Series:
    
    
    """
    Calculates percentage return based on contract positions and adjusted prices.
    """
    price_points = (adjusted_price - adjusted_price.shift(1)) * position_contracts_held.shift(1)
    instrument_currency_return = price_points * multiplier
    fx_aligned = fx_series.reindex(instrument_currency_return.index, method="ffill")
    base_currency_return = instrument_currency_return * fx_aligned
    pct_return = base_currency_return / capital_required
    return pct_return


#%% ============================
# Compute returns with costs

def calculate_perc_returns_with_costs(
    position_contracts_held: pd.Series,
    adjusted_price: pd.Series,
    fx_series: pd.Series,
    stdev_series: standardDeviation,
    multiplier: float,
    capital_required: float,
    cost_per_contract: float,
) -> pd.Series:

    precost_return_price_points = (
        adjusted_price - adjusted_price.shift(1)
    ) * position_contracts_held.shift(1)

    precost_return_instrument_currency = precost_return_price_points * multiplier
    historic_costs = calculate_costs_deflated_for_vol(
        stddev_series=stdev_series,
        cost_per_contract=cost_per_contract,
        position_contracts_held=position_contracts_held,
    )

    historic_costs_aligned = historic_costs.reindex(
        precost_return_instrument_currency.index, method="ffill"
    )
    return_instrument_currency = (
        precost_return_instrument_currency - historic_costs_aligned
    )

    fx_series_aligned = fx_series.reindex(
        return_instrument_currency.index, method="ffill"
    )
    return_base_currency = return_instrument_currency * fx_series_aligned

    perc_return = return_base_currency / capital_required

    return perc_return


def calculate_costs_deflated_for_vol(
    stddev_series: standardDeviation,
    cost_per_contract: float,
    position_contracts_held: pd.Series,
) -> pd.Series:

    round_position_contracts_held = position_contracts_held.round()
    position_change = (
        round_position_contracts_held - round_position_contracts_held.shift(1)
    )
    abs_trades = position_change.abs()

    historic_cost_per_contract = calculate_deflated_costs(
        stddev_series=stddev_series, cost_per_contract=cost_per_contract
    )

    historic_cost_per_contract_aligned = historic_cost_per_contract.reindex(
        abs_trades.index, method="ffill"
    )

    historic_costs = abs_trades * historic_cost_per_contract_aligned

    return historic_costs


def calculate_deflated_costs(
    stddev_series: standardDeviation, cost_per_contract: float
) -> pd.Series:

    stdev_daily_price = stddev_series.daily_risk_price_terms()

    final_stdev = stdev_daily_price[-1]
    cost_deflator = stdev_daily_price / final_stdev
    historic_cost_per_contract = cost_per_contract * cost_deflator

    return historic_cost_per_contract


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------
def calculate_stats(perc_return: pd.Series,
                    freq: Frequency = NATURAL) -> dict:
    """
    Computes key performance statistics such as annual mean, volatility,
    Sharpe ratio, skewness, drawdowns and quantile ratios.
    """
    return_freq = sum_by_frequency(perc_return, freq=freq)

    ann_mean = annualized_mean(return_freq, freq=freq)
    ann_std = annualized_std(return_freq, freq=freq)
    sharpe = ann_mean / ann_std

    skew = return_freq.skew()
    drawdowns = calculate_drawdown(return_freq)
    avg_drawdown = drawdowns.mean()
    max_drawdown = drawdowns.max()
    quant_lower = quantile_ratio_lower(return_freq)
    quant_upper = quantile_ratio_upper(return_freq)

    return dict(
        ann_mean=ann_mean,
        ann_std=ann_std,
        sharpe=sharpe,
        skew=skew,
        avg_drawdown=avg_drawdown,
        max_drawdown=max_drawdown,
        quant_lower=quant_lower,
        quant_upper=quant_upper
    )


#%% FX SERIES ADJUSTS   
# ============================

def create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict: dict, fx_series: dict) -> dict:
    """
    Create a dictionary of price series in base currency aligned with each instrument's adjusted price series.

    Parameters
    ----------
    adjusted_prices_dict : dict
        Dictionary where keys are instrument codes and values are adjusted price Series.

    Returns
    -------
    dict
        Dictionary where keys are instrument codes and values are FX rate Series aligned to the same dates.
    """
    fx_series_dict = dict(
        [
            (
                instrument_code,
                create_fx_series_given_adjusted_prices(
                    adjusted_prices=adjusted_prices, instrument_fx_serie=fx_series.get(instrument_code, pd.Series(1, index=adjusted_prices.index)),
                ),
            )
            for instrument_code, adjusted_prices in adjusted_prices_dict.items()
        ]
    )
    return fx_series_dict


def create_fx_series_given_adjusted_prices(
    adjusted_prices: pd.Series, 
    instrument_fx_serie: pd.Series | float,
) -> pd.Series:
    """
    Generate the price serie converted with FX rate for a given instrument based on its currency.

    Parameters
    ----------
    instrument_code : str
        Identifier of the financial instrument.
    adjusted_prices : pd.Series
        Adjusted price series of the instrument.
    instrument_currency : pd.Series
        FX rate for instrument to convert to the base currency
    

    Returns
    -------
    pd.Series
        FX rate series aligned to the instrument's adjusted price index.
    """

    fx_prices = instrument_fx_serie * adjusted_prices
    fx_prices_aligned = fx_prices.reindex(adjusted_prices.index).ffill()
    return fx_prices_aligned


# =============================
# CALCULATE POSITIONS WITH FUNCTION APPLIED FORECATS


def calculate_position_dict_with_forecast_from_function_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> dict:
    """
    Compute position series for all instruments by applying rule-based forecasts.

    Parameters
    ----------
    adjusted_prices_dict : dict
        Mapping {instrument_code: pd.Series} of adjusted prices.
    average_position_contracts_dict : dict
        Mapping {instrument_code: pd.Series} of long-run target positions (contracts).
    std_dev_dict : dict
        Mapping {instrument_code: standardDeviation} with volatility estimates.
    carry_prices_dict : dict
        Mapping {instrument_code: pd.Series} containing carry/roll-yield series.
    list_of_rules : list
        List of dictionaries. Each rule must define:
            - "function": callable that returns a forecast series
            - "scalar": float multiplier for the rule's output
            - additional rule arguments passed to the function

    Returns
    -------
    dict
        Mapping {instrument_code: pd.Series} with final contract positions.
    """
    list_of_instruments = list(adjusted_prices_dict.keys())

    position_dict_with_carry = dict(
        [
            (
                instrument_code,
                calculate_position_with_forecast_applied_from_function(
                    instrument_code,
                    average_position_contracts_dict=average_position_contracts_dict,
                    adjusted_prices_dict=adjusted_prices_dict,
                    std_dev_dict=std_dev_dict,
                    carry_prices_dict=carry_prices_dict,
                    list_of_rules=list_of_rules,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_carry


def calculate_position_with_forecast_applied_from_function(
    instrument_code: str,
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> pd.Series:
    """
    Compute the final scaled position for a single instrument using
    rule-defined forecasts.

    Parameters
    ----------
    instrument_code : str
        Identifier for the instrument.
    adjusted_prices_dict : dict
        Mapping {instrument_code: pd.Series} of adjusted prices.
    average_position_contracts_dict : dict
        Mapping {instrument_code: pd.Series} of long-run target position.
    std_dev_dict : dict
        Mapping {instrument_code: standardDeviation}.
    carry_prices_dict : dict
        Mapping {instrument_code: pd.Series} with carry/roll-yield data.
    list_of_rules : list
        Rules used to compute the combined forecast.

    Returns
    -------
    pd.Series
        Position series = combined_forecast * avg_position / 10.
    """
    forecast = calculate_combined_forecast_from_functions(
        instrument_code=instrument_code,
        adjusted_prices_dict=adjusted_prices_dict,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
        list_of_rules=list_of_rules,
    )

    return forecast * average_position_contracts_dict[instrument_code] / 10


def calculate_combined_forecast_from_functions(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> pd.Series:
    """
    Combine forecasts from multiple rule functions.

    Procedure:
        1. Evaluate each rule function.
        2. Concatenate their forecast series.
        3. Average across rules.
        4. Apply Forecast Diversification Multiplier (FDM).
        5. Clip final forecast to [-20, 20].

    Parameters
    ----------
    instrument_code : str
        Identifier of the instrument.
    adjusted_prices_dict : dict
        Dict of price series.
    std_dev_dict : dict
        Dict of volatility estimators per instrument.
    carry_prices_dict : dict
        Dict of carry/roll-yield series.
    list_of_rules : list
        Rules controlling forecast generation.

    Returns
    -------
    pd.Series
        Combined forecast series in standardized Carver-style units.
    """
    all_forecasts_as_list = [
        calculate_forecast_from_function(
            instrument_code=instrument_code,
            adjusted_prices_dict=adjusted_prices_dict,
            std_dev_dict=std_dev_dict,
            carry_prices_dict=carry_prices_dict,
            rule=rule,
        )
        for rule in list_of_rules
    ]

    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    rule_count = len(list_of_rules)
    fdm = get_fdm(rule_count)
    scaled_forecast = average_forecast * fdm

    capped_forecast = scaled_forecast.clip(-20, 20)
    return capped_forecast


def calculate_forecast_from_function(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    rule: dict,
) -> pd.Series:
    """
    Evaluate a single rule function and scale its output.

    Parameters
    ----------
    instrument_code : str
        Instrument identifier.
    adjusted_prices_dict : dict
        Dict mapping instrument → adjusted price series.
    std_dev_dict : dict
        Dict mapping instrument → volatility estimator.
    carry_prices_dict : dict
        Dict mapping instrument → carry series.
    rule : dict
        Rule specification. Must contain:
            - "function": callable
            - "scalar": float
            - Any additional arguments

    Returns
    -------
    pd.Series
        Scaled forecast series = function_output * scalar.
    """
    rule_copy = copy(rule)
    rule_function = rule_copy.pop("function")
    scalar = rule_copy.pop("scalar")
    rule_args = rule_copy

    forecast_value = rule_function(
        instrument_code=instrument_code,
        adjusted_prices_dict=adjusted_prices_dict,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
        **rule_args
    )

    return forecast_value * scalar


# Buffering positions

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


# ============================
# CORE FUNCTIONS FOR FAST DIRECTIONAL STRATEGIES


FORECAST_SCALAR = 9.3
AVG_ABS_FORECAST = 10.0

# ======================================================
#  ORDER STRUCTURES (GENERIC)
# ======================================================

OrderType = Enum("OrderType", ["LIMIT", "MARKET"])


@dataclass
class Order:
    """
    Generic order object used by any trading simulation.
    """
    order_type: OrderType
    qty: int
    limit_price: float = np.nan

    @property
    def is_buy(self):
        return self.qty > 0

    @property
    def is_sell(self):
        return self.qty < 0


class ListOfOrders(list):
    """
    Simple list of Order objects with helper filters.
    """
    def __init__(self, list_of_orders):
        super().__init__(list_of_orders)

    def drop_buy_limits(self):
        return self.drop_signed_limit_orders(1)

    def drop_sell_limits(self):
        return self.drop_signed_limit_orders(-1)

    def drop_signed_limit_orders(self, order_sign):
        return ListOfOrders([
            order
            for order in self
            if true_if_order_is_market_or_order_is_not_of_sign(order, order_sign)
        ])


def true_if_order_is_market_or_order_is_not_of_sign(order, order_sign_to_drop):
    """
    Used for filtering buy/sell limit orders.
    """
    if order.order_type == OrderType.MARKET:
        return True
    return np.sign(order.qty) != order_sign_to_drop


# ======================================================
#  TRADE EXECUTION (GENERIC)
# ======================================================

@dataclass
class Trade:
    qty: int
    fill_date: datetime.datetime
    current_price: float = np.nan

    @property
    def filled(self):
        return not self.unfilled

    @property
    def unfilled(self):
        return self.qty == 0


def fill_list_of_orders(list_of_orders, fill_date, current_price, bid_ask_spread):
    """
    Execute all orders at the given time. Only one order can be filled.
    """
    trades = [
        fill_order(
            order,
            current_price=current_price,
            fill_date=fill_date,
            bid_ask_spread=bid_ask_spread,
        )
        for order in list_of_orders
    ]

    trades = [t for t in trades if t.filled]

    if len(trades) == 0:
        return Trade(qty=0, fill_date=fill_date, current_price=current_price)
    if len(trades) == 1:
        return trades[0]

    raise Exception("Impossible for multiple trades to be filled at same time!")


def fill_order(order, current_price, fill_date, bid_ask_spread):
    """
    Fill any order respecting type and fill rules.
    """
    if order.order_type == OrderType.MARKET:
        return fill_market_order(order, current_price, fill_date, bid_ask_spread)
    elif order.order_type == OrderType.LIMIT:
        return fill_limit_order(order, fill_date, current_price)
    raise Exception("Order type not recognized")


def fill_market_order(order, current_price, fill_date, bid_ask_spread):
    """
    Market orders execute immediately at price ± spread.
    """
    if order.is_buy:
        exec_price = current_price + bid_ask_spread
    elif order.is_sell:
        exec_price = current_price - bid_ask_spread
    else:
        return Trade(qty=0, fill_date=fill_date, current_price=current_price)

    return Trade(qty=order.qty, fill_date=fill_date, current_price=exec_price)


def fill_limit_order(order, fill_date, current_price):
    """
    Limit orders execute only if price crosses the limit.
    """
    if order.is_buy and current_price > order.limit_price:
        return Trade(qty=0, fill_date=fill_date, current_price=current_price)

    if order.is_sell and current_price < order.limit_price:
        return Trade(qty=0, fill_date=fill_date, current_price=current_price)

    return Trade(
        qty=order.qty,
        fill_date=fill_date,
        current_price=order.limit_price,
    )


# ======================================================
#  PNL COMPUTATION (GENERIC)
# ======================================================

def calculate_perc_returns_from_trade_list(
    list_of_trades,
    multiplier,
    capital,
    fx_series,
    current_price_series,
    commission_per_contract,
    daily_stdev,
):
    """
    Convert a list of executed trades into a percentage return series.

    Parameters
    ----------
    list_of_trades : list of Trade
    multiplier : float
    capital : float
    fx_series : pd.Series
    current_price_series : pd.Series
    commission_per_contract : float
    daily_stdev : pd.Series or standardDeviation

    Returns
    -------
    pd.Series
    """
    
    
    # Ajustar essa função para estratégia 26
    
    qty_list = [t.qty for t in list_of_trades]
    date_list = [t.fill_date for t in list_of_trades]
    price_list = [t.current_price for t in list_of_trades]

    qty_series = pd.Series(qty_list, index=date_list)
    price_series = pd.Series(price_list, index=date_list)
    position_series = qty_series.cumsum()

    return calculate_perc_returns_with_costs(
        position_contracts_held=position_series,
        adjusted_price=price_series,
        fx_series=fx_series,
        capital_required=capital,
        multiplier=multiplier,
        cost_per_contract=commission_per_contract,
        stdev_series=daily_stdev,
    )
