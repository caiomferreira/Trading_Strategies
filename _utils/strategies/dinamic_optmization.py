import pandas as pd
import numpy as np
from _utils.core_functions import *
from _utils.core_extended_functions import *
from _utils.correlation_estimate import *

from _utils.strategies.trend_asset_classes import (
    calculate_asset_class_price_dict,
    calculate_relative_price_dict
    )


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

def dynamically_optimise_positions(
    capital: float,
    fx_series_dict: dict,
    unrounded_position_contracts_dict: dict,
    multipliers: dict,
    std_dev_dict: dict,
    current_prices_dict: dict,
    adjusted_prices_dict: dict,
    cost_per_contract_dict: dict,
    algo_to_use,
) -> dict:
    """
    Run dynamic position optimisation across time.

    Method:
        1. Aggregate all relevant data into a single optimisation container.
        2. Loop through each date in the common index.
        3. For each date, extract data for that period.
        4. Optimise contracts using chosen optimisation algorithm.
        5. Maintain previous positions to ensure path-dependence.
        6. Convert the final DataFrame to a dict of Series.

    Parameters
    ----------
    capital : float
        Total capital available.
    fx_series_dict : dict
        FX series mapping {instrument: fx_rate_series}.
    unrounded_position_contracts_dict : dict
        Initial unrounded position targets by instrument.
    multipliers : dict
        Contract multipliers per instrument.
    std_dev_dict : dict
        Standard deviation objects for each instrument.
    current_prices_dict : dict
        Mapping {instrument: current price series}.
    adjusted_prices_dict : dict
        Adjusted prices for volatility scaling and returns.
    cost_per_contract_dict : dict
        Cost per contract for each instrument.
    algo_to_use : callable
        Optimisation algorithm to use on each period.

    Returns
    -------
    dict
        Mapping {instrument: pd.Series} of dynamically optimised contract positions.
    """
    data_for_optimisation = get_data_for_dynamic_optimisation(
        capital=capital,
        current_prices_dict=current_prices_dict,
        std_dev_dict=std_dev_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        fx_series_dict=fx_series_dict,
        adjusted_prices_dict=adjusted_prices_dict,
        multipliers=multipliers,
        unrounded_position_contracts_dict=unrounded_position_contracts_dict,
    )

    position_list = []
    common_index = data_for_optimisation.common_index
    previous_position = get_initial_positions(unrounded_position_contracts_dict)

    for relevant_date in common_index:
        data_for_single_period = get_data_for_relevant_date(
            relevant_date, data_for_optimisation=data_for_optimisation
        )

        optimal_positions = optimisation_for_single_period(
            previous_position=previous_position,
            data_for_single_period=data_for_single_period,
            algo_to_use=algo_to_use,
        )

        position_list.append(optimal_positions)
        previous_position = copy(optimal_positions)

    position_df = pd.DataFrame(position_list, index=common_index)
    positions_as_dict = from_df_to_dict_of_series(position_df)

    return positions_as_dict


def optimisation_for_single_period(
    previous_position: positionContracts,
    data_for_single_period: dataForSinglePeriod,
    algo_to_use,
) -> positionContracts:
    """
    Optimise positions for a single date, handling missing data and asset filtering.

    Steps:
        1. Determine which assets have valid data.
        2. Remove assets without data.
        3. Restrict previous positions to valid assets.
        4. Optimise positions for the remaining assets.
        5. Re-insert missing assets as filled positions.

    Parameters
    ----------
    previous_position : positionContracts
        Previous day's contract positions.
    data_for_single_period : dataForSinglePeriod
        Data container for a specific date.
    algo_to_use : callable
        Optimisation routine.

    Returns
    -------
    positionContracts
        Full set of optimised positions for the period.
    """
    assets_with_data = which_assets_have_data(data_for_single_period)
    if len(assets_with_data) == 0:
        return previous_position

    assets_without_data = which_assets_without_data(
        data_for_single_period, assets_with_data=assets_with_data
    )

    data_for_single_period = data_for_single_period_with_valid_assets_only(
        data_for_single_period, assets_with_data=assets_with_data
    )

    previous_position = previous_position.with_selected_assets_only(assets_with_data)

    optimised_position = optimisation_for_single_period_with_valid_assets_only(
        previous_position=previous_position,
        data_for_single_period=data_for_single_period,
        algo_to_use=algo_to_use,
    )

    optimised_position_with_all_assets = (
        optimised_position.with_fill_for_missing_assets(assets_without_data)
    )

    return optimised_position_with_all_assets


def optimisation_for_single_period_with_valid_assets_only(
    previous_position: positionContracts,
    data_for_single_period: dataForSinglePeriod,
    algo_to_use,
) -> positionContracts:
    """
    Optimise positions ignoring assets lacking data.

    Workflow:
        - Convert data_for_single_period to weight-based structure.
        - Run optimisation to get optimal weights.
        - Convert weights → contract counts.

    Parameters
    ----------
    previous_position : positionContracts
        Previous contract positions, restricted to valid assets.
    data_for_single_period : dataForSinglePeriod
        Period data filtered for valid assets.
    algo_to_use : callable
        Optimisation algorithm.

    Returns
    -------
    positionContracts
        Optimised contract positions.
    """
    data_for_single_period_with_weights = (
        dataForSinglePeriodWithWeights.from_data_for_single_period(
            previous_position=previous_position,
            data_for_single_period=data_for_single_period,
        )
    )

    optimised_weights = optimisation_of_weight_for_single_period_with_valid_assets_only(
        data_for_single_period_with_weights, algo_to_use=algo_to_use
    )

    weights_per_contract = data_for_single_period_with_weights.weight_per_contract
    optimised_contracts = position_contracts_from_position_weights(
        optimised_weights, weights_per_contract=weights_per_contract
    )

    return optimised_contracts


def optimisation_of_weight_for_single_period_with_valid_assets_only(
    data_for_single_period_with_weights: dataForSinglePeriodWithWeights, algo_to_use
) -> positionWeights:
    """
    Optimise weights for assets with valid data only.

    Converts the structured data to a Numpy representation,
    runs the optimisation algorithm, then converts results
    back to a positionWeights object.

    Parameters
    ----------
    data_for_single_period_with_weights : dataForSinglePeriodWithWeights
        Structured period data.
    algo_to_use : callable
        Optimisation algorithm (e.g., greedy search).

    Returns
    -------
    positionWeights
        Optimal weight vector.
    """
    data_for_single_period_as_np = (
        dataForSinglePeriodWithWeightsAsNp.from_data_for_single_period_with_weights(
            data_for_single_period_with_weights
        )
    )

    solution_as_np = algo_to_use(data_for_single_period_as_np)

    list_of_assets = list(
        data_for_single_period_with_weights.unrounded_optimal_position_weights.keys()
    )

    solution_as_weights = positionWeights.from_weights_and_keys(
        list_of_keys=list_of_assets, list_of_weights=list(solution_as_np)
    )

    return solution_as_weights


def greedy_algo_across_integer_values(
    data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp,
) -> np.array:
    """
    Greedy optimisation algorithm that iteratively searches integer weight moves
    that reduce tracking error relative to target weights.

    Method:
        - Start from initial weights.
        - For each iteration, try incrementing each asset by +/- 1 contract
          (scaled by weight_per_contract).
        - Accept moves that reduce tracking error.
        - Stop when no improvement is possible.

    Parameters
    ----------
    data_for_single_period_as_np : dataForSinglePeriodWithWeightsAsNp
        Numpy-structured optimisation input.

    Returns
    -------
    np.array
        Optimised weight vector.
    """
    weight_start = data_for_single_period_as_np.starting_weights

    current_best_value = evaluate_tracking_error(
        weight_start, data_for_single_period_as_np
    )
    current_best_solution = weight_start

    while True:
        new_best_value, new_best_solution = find_best_proposed_solution(
            current_best_solution=current_best_solution,
            current_best_value=current_best_value,
            data_for_single_period_as_np=data_for_single_period_as_np,
        )
        if new_best_value < current_best_value:
            current_best_value = new_best_value
            current_best_solution = new_best_solution
        else:
            break

    return current_best_solution


def evaluate_tracking_error(
    weights: np.array,
    data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp,
):
    """
    Evaluate tracking error between current weights and target weights.

    Parameters
    ----------
    weights : np.array
        Proposed weight vector.
    data_for_single_period_as_np : dataForSinglePeriodWithWeightsAsNp
        Numpy-structured data holding target weights & covariance matrix.

    Returns
    -------
    float
        Tracking error (standard deviation).
    """
    optimal_weights = data_for_single_period_as_np.unrounded_optimal_position_weights
    covariance = data_for_single_period_as_np.covariance_matrix

    return evaluate_tracking_error_for_weights(
        weights, optimal_weights, covariance=covariance
    )


def evaluate_tracking_error_for_weights(
    weights: np.array, other_weights, covariance: np.array
) -> float:
    """
    Compute tracking error (std) between two weight vectors.

    Parameters
    ----------
    weights : np.array
        Candidate weight vector.
    other_weights : np.array
        Target/unrounded weights.
    covariance : np.array
        Covariance matrix of returns.

    Returns
    -------
    float
        Standard deviation of tracking error.

    Raises
    ------
    Exception
        If covariance * gap produces a negative variance.
    """
    solution_gap = weights - other_weights
    track_error_var = solution_gap.dot(covariance).dot(solution_gap)

    if track_error_var < 0:
        raise Exception("Negative covariance when optimising!")

    return track_error_var ** 0.5


def find_best_proposed_solution(
    current_best_solution: np.array,
    current_best_value: float,
    data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp,
) -> tuple:
    """
    Try all one-step weight adjustments and choose the one that most reduces TE.

    Each asset is tested for:
        weight[i] ± weight_per_contract[i] * direction[i]

    Parameters
    ----------
    current_best_solution : np.array
        Current best weight vector.
    current_best_value : float
        Current best tracking error.
    data_for_single_period_as_np : dataForSinglePeriodWithWeightsAsNp
        Structured data for optimisation.

    Returns
    -------
    tuple
        (best_value, best_solution) after exploring all neighbours.
    """
    best_proposed_value = copy(current_best_value)
    best_proposed_solution = copy(current_best_solution)

    per_contract_value = data_for_single_period_as_np.weight_per_contract
    direction = data_for_single_period_as_np.direction_as_np

    count_assets = len(best_proposed_solution)

    for i in range(count_assets):
        incremented_solution = copy(current_best_solution)
        incremented_solution[i] = (
            incremented_solution[i] + per_contract_value[i] * direction[i]
        )

        incremented_objective_value = evaluate_tracking_error(
            incremented_solution, data_for_single_period_as_np
        )

        if incremented_objective_value < best_proposed_value:
            best_proposed_value = incremented_objective_value
            best_proposed_solution = incremented_solution

    return best_proposed_value, best_proposed_solution
