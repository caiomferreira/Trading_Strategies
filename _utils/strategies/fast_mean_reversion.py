import numpy as np


from _utils.core_functions import (
    Order, OrderType, ListOfOrders,
    fill_list_of_orders,
    calculate_perc_returns_from_trade_list,
)
from _utils.portfoliohandcrafiting import *

from _utils.correlation_estimate import get_row_of_series_before_date

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


FORECAST_SCALAR = 9.3
AVG_ABS_FORECAST = 10.0


# ======================================================
#  TOP-LEVEL: PnL PARA TODOS INSTRUMENTOS
# ======================================================

def generate_pandl_across_instruments_for_hourly_data(
    adjusted_prices_daily_dict,
    current_prices_daily_dict,
    adjusted_prices_hourly_dict,
    std_dev_dict,
    average_position_contracts_dict,
    fx_series_dict,
    multipliers,
    commission_per_contract_dict,
    capital,
    tick_size_dict,
    bid_ask_spread_dict,
    trade_calculation_function,
):
    """
    Compute MR-PnL for all instruments.

    Returns
    -------
    dict {instrument: pd.Series}
    """
    instr_list = list(adjusted_prices_hourly_dict.keys())

    return {
        instr: calculate_pandl_series_for_instrument(
            adjusted_daily_prices=adjusted_prices_daily_dict[instr],
            current_daily_prices=current_prices_daily_dict[instr],
            adjusted_hourly_prices=adjusted_prices_hourly_dict[instr],
            daily_stdev=std_dev_dict[instr],
            average_position_daily=average_position_contracts_dict[instr],
            fx_series=fx_series_dict[instr],
            multiplier=multipliers[instr],
            commission_per_contract=commission_per_contract_dict[instr],
            tick_size=tick_size_dict[instr],
            bid_ask_spread=bid_ask_spread_dict[instr],
            capital=capital,
            instrument_code=instr,
            trade_calculation_function=trade_calculation_function,
        )
        for instr in instr_list
    }


# ======================================================
#  PNL PARA 1 INSTRUMENTO
# ======================================================

def calculate_pandl_series_for_instrument(
    adjusted_daily_prices,
    current_daily_prices,
    adjusted_hourly_prices,
    daily_stdev,
    average_position_daily,
    fx_series,
    multiplier,
    capital,
    tick_size,
    bid_ask_spread,
    commission_per_contract,
    instrument_code,
    trade_calculation_function,
):
    """
    Build the MR-PnL for a single instrument.

    Returns
    -------
    pd.Series (percentage returns)
    """
    trades = generate_list_of_mr_trades_for_instrument(
        adjusted_daily_prices=adjusted_daily_prices,
        current_daily_prices=current_daily_prices,
        adjusted_hourly_prices=adjusted_hourly_prices,
        daily_stdev=daily_stdev,
        average_position_daily=average_position_daily,
        tick_size=tick_size,
        bid_ask_spread=bid_ask_spread,
        instrument_code=instrument_code,
        trade_calculation_function=trade_calculation_function,
    )

    return calculate_perc_returns_from_trade_list(
        list_of_trades=trades,
        capital=capital,
        fx_series=fx_series,
        commission_per_contract=commission_per_contract,
        current_price_series=current_daily_prices,
        multiplier=multiplier,
        daily_stdev=daily_stdev,
    )


# ======================================================
#  GERAÇÃO DE TRADES MR
# ======================================================

def generate_list_of_mr_trades_for_instrument(
    adjusted_daily_prices,
    current_daily_prices,
    adjusted_hourly_prices,
    daily_stdev,
    average_position_daily,
    tick_size,
    bid_ask_spread,
    instrument_code,
    trade_calculation_function,
):
    """
    Generate trades for MR strategy.

    Steps:
    1) Compute equilibrium projected to hourly
    2) Compute sigma_p hourly
    3) Build trade list via trade_calculation_function
    """
    daily_eq_hourly = calculate_equilibrium(
        adjusted_daily_prices=adjusted_daily_prices,
        adjusted_hourly_prices=adjusted_hourly_prices,
    )

    hourly_sigma = calculate_sigma_p(
        current_daily_prices=current_daily_prices,
        daily_stdev=daily_stdev,
        adjusted_hourly_prices=adjusted_hourly_prices,
    )

    return calculate_trades_for_instrument(
        adjusted_hourly_prices=adjusted_hourly_prices,
        daily_equilibrium_hourly=daily_eq_hourly,
        average_position_daily=average_position_daily,
        hourly_stdev_prices=hourly_sigma,
        bid_ask_spread=bid_ask_spread,
        tick_size=tick_size,
        instrument_code=instrument_code,
        trade_calculation_function=trade_calculation_function,
    )


def calculate_trades_for_instrument(
    adjusted_hourly_prices,
    daily_equilibrium_hourly,
    average_position_daily,
    hourly_stdev_prices,
    bid_ask_spread,
    tick_size,
    instrument_code,
    trade_calculation_function,
):
    """
    Core engine of MR trade generation.
    """
    trades = []
    orders = ListOfOrders([])
    current_position = 0
    dates = list(adjusted_hourly_prices.index)

    for dt in dates[1:]:
        price = float(get_row_of_series_before_date(adjusted_hourly_prices, dt))
        if np.isnan(price):
            continue

        trade = fill_list_of_orders(orders, dt, price, bid_ask_spread)
        
        
        if trade.filled:
            trades.append(trade)
            current_position += trade.qty

        eq = get_row_of_series_before_date(daily_equilibrium_hourly, dt)
        avg = get_row_of_series_before_date(average_position_daily, dt)
        sigma = get_row_of_series_before_date(hourly_stdev_prices, dt)

        orders = trade_calculation_function(
            current_position=current_position,
            current_price=price,
            current_equilibrium=eq,
            current_average_position=avg,
            current_hourly_stdev_price=sigma,
            tick_size=tick_size,
            instrument_code=instrument_code,
            relevant_date=dt,
        )
        

    return trades


# ======================================================
#  MR ORDER GENERATION
# ======================================================

def required_orders_for_mr_system(
    current_position,
    current_equilibrium,
    current_hourly_stdev_price,
    current_price,
    current_average_position,
    tick_size,
    instrument_code,
    relevant_date,
):
    """
    MR-specific rule for required orders.
    """
    forecast = mr_forecast_unclipped(
        current_equilibrium=current_equilibrium,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_price=current_price,
    )

    orders = calculate_orders_given_forecast_and_positions(
        current_average_position=current_average_position,
        current_forecast=forecast,
        current_equilibrium=current_equilibrium,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_position=current_position,
        tick_size=tick_size,
    )

    if forecast < -20:
        return orders.drop_sell_limits()
    if forecast > 20:
        return orders.drop_buy_limits()

    return orders


def calculate_orders_given_forecast_and_positions(
    current_forecast,
    current_position,
    current_equilibrium,
    current_hourly_stdev_price,
    current_average_position,
    tick_size,
):
    """
    Main decision rule for MR order generation.
    """
    target_position = optimal_position_given_unclipped_forecast(
        current_forecast=current_forecast,
        current_average_position=current_average_position,
    )

    delta = int(np.round(target_position - current_position))

    if abs(delta) > 1:
        return ListOfOrders([Order(OrderType.MARKET, qty=delta)])

    buy_limit = get_limit_price_given_resulting_position_with_tick_size_applied(
        number_of_contracts_to_solve_for=current_position + 1,
        current_equilibrium=current_equilibrium,
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        tick_size=tick_size,
    )

    sell_limit = get_limit_price_given_resulting_position_with_tick_size_applied(
        number_of_contracts_to_solve_for=current_position - 1,
        current_equilibrium=current_equilibrium,
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        tick_size=tick_size,
    )

    return ListOfOrders([
        Order(OrderType.LIMIT, qty=1, limit_price=buy_limit),
        Order(OrderType.LIMIT, qty=-1, limit_price=sell_limit),
    ])


# ======================================================
#  MR FORECAST / EQUILIBRIUM / SIGMA
# ======================================================

def mr_forecast_unclipped(current_equilibrium, current_hourly_stdev_price, current_price):
    """
    Compute raw MR forecast before clipping.
    """
    raw = current_equilibrium - current_price
    risk_adj = raw / current_hourly_stdev_price
    return risk_adj * FORECAST_SCALAR


def optimal_position_given_unclipped_forecast(current_forecast, current_average_position):
    """
    Convert MR forecast into target position.
    """
    clipped = np.clip(current_forecast, -20, 20)
    return clipped * current_average_position / AVG_ABS_FORECAST


def get_limit_price_given_resulting_position_with_tick_size_applied(
    number_of_contracts_to_solve_for,
    current_equilibrium,
    current_hourly_stdev_price,
    current_average_position,
    tick_size,
):
    """
    Compute theoretical limit price then round to nearest tick.
    """
    raw = get_limit_price_given_resulting_position(
        number_of_contracts_to_solve_for=number_of_contracts_to_solve_for,
        current_equilibrium=current_equilibrium,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_average_position=current_average_position,
    )

    return np.round(raw / tick_size) * tick_size


def get_limit_price_given_resulting_position(
    number_of_contracts_to_solve_for,
    current_equilibrium,
    current_hourly_stdev_price,
    current_average_position,
):
    """
    Theoretical limit price for MR system.
    """
    return current_equilibrium - (
        number_of_contracts_to_solve_for
        * AVG_ABS_FORECAST
        * current_hourly_stdev_price
        / (FORECAST_SCALAR * current_average_position)
    )


def calculate_equilibrium(adjusted_daily_prices, adjusted_hourly_prices):
    """
    Equilibrium = EWMA(5 days) of adjusted_daily_prices projected to hourly.
    """
    daily_eq = adjusted_daily_prices.ewm(5).mean()
    return daily_eq.reindex(adjusted_hourly_prices.index, method="ffill")


def calculate_sigma_p(current_daily_prices, daily_stdev, adjusted_hourly_prices):
    """
    Convert daily stdev to hourly stdev-price.
    """
    stdev_prices_daily = daily_stdev * current_daily_prices / 16
    return stdev_prices_daily.reindex(adjusted_hourly_prices.index, method="ffill")


def generate_mr_forecast_series_for_instrument(
    daily_equilibrium_hourly,
    adjusted_hourly_prices,
    hourly_stdev_prices,
):
    """
    MR forecast series (vectorized).
    """
    raw = daily_equilibrium_hourly - adjusted_hourly_prices.squeeze()
    risk_adj = raw / hourly_stdev_prices
    scaled = risk_adj * FORECAST_SCALAR
    return scaled.clip(-20, 20)
