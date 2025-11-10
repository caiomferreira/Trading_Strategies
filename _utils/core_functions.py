import pandas as pd
import numpy as np
from enum import Enum
from scipy.stats import norm
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

# --------------------------------------------------
# FREQUENCY HANDLING
# --------------------------------------------------
PERIODS_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "yearly": 1}
SECONDS_PER_YEAR = 365 * 24 * 60 * 60


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
                           multiplier: pd.Series,
                           fx_series: float,
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


