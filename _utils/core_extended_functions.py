import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime

from _utils.core_functions import *
from _utils.correlation_estimate import *


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

class genericValuesPerContract(dict):
    """
    Generic key→value container for per-contract quantities (weights, costs, exposures).

    Behaves like a specialized dict, with helper constructors for:
        - building all-zero dicts
        - building constant-value dicts
        - building dicts from aligned lists of keys and weights
    """

    @classmethod
    def allzeros(cls, list_of_keys: list):
        """
        Create a genericValuesPerContract object where all keys map to 0.0.
        """
        return cls.all_one_value(list_of_keys, value=0.0)

    @classmethod
    def all_one_value(cls, list_of_keys: list, value=0.0):
        """
        Create an object where every key maps to the same provided value.
        """
        return cls.from_weights_and_keys(
            list_of_weights=[value] * len(list_of_keys),
            list_of_keys=list_of_keys
        )

    @classmethod
    def from_weights_and_keys(cls, list_of_weights: list, list_of_keys: list):
        """
        Build object from parallel lists of keys and weights.

        Raises
        ------
        AssertionError
            If lengths do not match.
        """
        assert len(list_of_keys) == len(list_of_weights)
        pweights_as_list = [(key, weight)
                            for key, weight in zip(list_of_keys, list_of_weights)]
        return cls(pweights_as_list)


class positionContracts(genericValuesPerContract):
    """
    Represents contract positions for a set of assets.

    Includes helper functions for filtering assets and filling missing assets.
    """

    def with_selected_assets_only(self, assets_with_data: list):
        """
        Keep only positions for assets that have valid data.

        Returns
        -------
        positionContracts
        """
        as_dict_with_selected_assets = dict_with_selected_assets_only(
            some_dict=self,
            assets_with_data=assets_with_data
        )
        return positionContracts(as_dict_with_selected_assets)

    def with_fill_for_missing_assets(
        self,
        missing_assets: list,
        fill_value: float = 0.0
    ):
        """
        Add missing assets with a default fill_value.

        Returns
        -------
        positionContracts
        """
        new_dict = {key: fill_value for key in missing_assets}
        joint_dict = {**self, **new_dict}
        return positionContracts(joint_dict)


def dict_with_selected_assets_only(some_dict: dict, assets_with_data: list) -> dict:
    """
    Filter a dictionary to contain only a selected subset of keys.
    """
    return {key: some_dict[key] for key in assets_with_data}


@dataclass
class dataForOptimisation:
    """
    Container aggregating all data needed for multi-period dynamic optimisation.
    """
    common_index: list
    list_of_covariance_matrices: covarianceList
    deflated_costs_dict: dict
    unrounded_position_contracts_dict: dict
    fx_series_dict: dict
    multipliers: dict
    capital: float
    current_prices_dict: dict


@dataclass
class dataForSinglePeriod:
    """
    Data structure for a single date's optimisation input:
        - contract multipliers
        - FX rates
        - target (unrounded) positions
        - covariance matrix
        - capital and cost information
    """
    multipliers: dict
    current_prices_this_period: dict
    unrounded_optimal_positions: positionContracts
    fx_rates_this_period: dict
    capital: float
    covariance_this_period: covarianceEstimate
    current_cost_per_contract: dict


class weightPerContract(genericValuesPerContract):
    """
    Per-contract weight (exposure divided by capital).
    """

    @classmethod
    def from_single_period_data(cls, data_for_single_period: dataForSinglePeriod):
        """
        Compute weight_per_contract = notional_exposure_per_contract / capital.
        """
        notional_exposure_per_contract = (
            notionalExposurePerContract.from_single_period_data(data_for_single_period)
        )
        weight_per_contract_dict = divide_dict_by_float_dict(
            notional_exposure_per_contract,
            data_for_single_period.capital
        )
        return cls(weight_per_contract_dict)


class notionalExposurePerContract(genericValuesPerContract):
    """
    Notional exposure per contract = FX * multiplier * current_price.
    """

    @classmethod
    def from_single_period_data(cls, data_for_single_period: dataForSinglePeriod):
        """
        Compute the product:
            FX_rate * multiplier * current_price
        for each instrument.
        """
        multiplier_as_dict_base_fx = multiplied_dict(
            data_for_single_period.fx_rates_this_period,
            data_for_single_period.multipliers,
        )
        notional_exposure_as_dict = multiplied_dict(
            multiplier_as_dict_base_fx,
            data_for_single_period.current_prices_this_period,
        )
        return cls(notional_exposure_as_dict)


class positionWeights(genericValuesPerContract):
    """
    Position weights expressed as fraction of capital.
    """

    @classmethod
    def from_positions_and_weight_per_contract(
        cls,
        positions: positionContracts,
        weights_per_contract: weightPerContract
    ):
        """
        Convert contract positions → weights:
            weight = position * weight_per_contract
        """
        position_weights_as_dict = multiplied_dict(positions, weights_per_contract)
        return cls(position_weights_as_dict)


class costsAsWeights(genericValuesPerContract):
    """
    Trading costs expressed in weight terms relative to capital.
    """

    @classmethod
    def from_costs_capital_and_weight_per_contract(
        cls,
        costs_in_base_currency: dict,
        capital: float,
        weights_per_contract: weightPerContract,
    ):
        """
        Convert costs → weight terms:
            cost_weight = (cost / capital) / weight_per_contract
        """
        costs_as_proportion_of_capital = divide_dict_by_float_dict(
            costs_in_base_currency,
            capital
        )
        costs_in_weight_terms = divided_dict(
            costs_as_proportion_of_capital,
            weights_per_contract
        )
        return cls(costs_in_weight_terms)


def position_contracts_from_position_weights(
    position_weights: positionWeights,
    weights_per_contract: weightPerContract
) -> positionContracts:
    """
    Convert weights back into contract counts:
        contracts = weight / weight_per_contract
    """
    position_contracts_as_dict = divided_dict(
        position_weights,
        weights_per_contract
    )
    return positionContracts(position_contracts_as_dict)


def multiplied_dict(dicta, dictb):
    """
    Elementwise multiplication of two dicts with matching keys.
    """
    return {key: value * dictb[key] for key, value in dicta.items()}


def divided_dict(dicta, dictb):
    """
    Elementwise division of two dicts with matching keys.
    """
    return {key: value / dictb[key] for key, value in dicta.items()}


def divide_dict_by_float_dict(dicta, floatb):
    """
    Divide all values in dict by a float.
    """
    return {key: value / floatb for key, value in dicta.items()}


@dataclass
class dataForSinglePeriodWithWeights:
    """
    Full data structure for a single time step, including:
        - weight per contract
        - previous and target weights
        - covariance matrix
        - trading costs in weight terms
    """
    weight_per_contract: weightPerContract
    previous_position_weights: positionWeights
    unrounded_optimal_position_weights: positionWeights
    covariance_matrix: covarianceEstimate
    costs_in_weight_terms: costsAsWeights

    @classmethod
    def from_data_for_single_period(
        cls,
        previous_position: positionContracts,
        data_for_single_period: dataForSinglePeriod,
    ):
        """
        Build structure by converting positions and costs to weight-based objects.
        """
        weight_per_contract = weightPerContract.from_single_period_data(
            data_for_single_period
        )

        previous_position_weights = positionWeights.from_positions_and_weight_per_contract(
            positions=previous_position,
            weights_per_contract=weight_per_contract
        )

        unrounded_optimal_position_weights = (
            positionWeights.from_positions_and_weight_per_contract(
                positions=data_for_single_period.unrounded_optimal_positions,
                weights_per_contract=weight_per_contract,
            )
        )

        costs_as_weights = costsAsWeights.from_costs_capital_and_weight_per_contract(
            capital=data_for_single_period.capital,
            weights_per_contract=weight_per_contract,
            costs_in_base_currency=data_for_single_period.current_cost_per_contract,
        )

        return cls(
            weight_per_contract,
            previous_position_weights,
            unrounded_optimal_position_weights,
            data_for_single_period.covariance_this_period,
            costs_as_weights,
        )


@dataclass
class dataForSinglePeriodWithWeightsAsNp:
    """
    Numpy-ready version of dataForSinglePeriodWithWeights for faster optimisation.
    Contains:
        - covariance matrix
        - target weights
        - previous weights
        - weight_per_contract
        - direction vector
        - cost vector
    """
    covariance_matrix: np.array
    unrounded_optimal_position_weights: np.array
    previous_position_weights: np.array
    weight_per_contract: np.array
    starting_weights: np.array
    direction_as_np: np.array
    cost_in_weight_terms_as_np: np.array

    @classmethod
    def from_data_for_single_period_with_weights(
        cls,
        data_for_single_period_with_weights: dataForSinglePeriodWithWeights
    ):
        """
        Convert dict-based weight structures to numpy vectors for optimisation.
        """
        unrounded_optimal_position_weights_as_np = np.array(
            list(data_for_single_period_with_weights
                 .unrounded_optimal_position_weights.values())
        )
        previous_position_weights_as_np = np.array(
            list(data_for_single_period_with_weights
                 .previous_position_weights.values())
        )
        weight_per_contract_as_np = np.array(
            list(data_for_single_period_with_weights
                 .weight_per_contract.values())
        )
        direction_as_np = np.sign(unrounded_optimal_position_weights_as_np)
        covariance_as_np = data_for_single_period_with_weights.covariance_matrix.values

        starting_weights = zero_np_weights_given_direction_as_np(direction_as_np)

        cost_in_weight_terms_as_np = np.array(
            list(data_for_single_period_with_weights
                 .costs_in_weight_terms.values())
        )

        return cls(
            covariance_as_np,
            unrounded_optimal_position_weights_as_np,
            previous_position_weights_as_np,
            weight_per_contract_as_np,
            starting_weights,
            direction_as_np,
            cost_in_weight_terms_as_np,
        )


def zero_np_weights_given_direction_as_np(direction_as_np: np.array) -> np.array:
    """
    Initialise starting weight vector as all zeros.
    """
    return np.array([0.0] * len(direction_as_np))


def get_data_for_dynamic_optimisation(
    capital: float,
    fx_series_dict: dict,
    unrounded_position_contracts_dict: dict,
    multipliers: dict,
    std_dev_dict: dict,
    current_prices_dict: dict,
    adjusted_prices_dict: dict,
    cost_per_contract_dict: dict,
) -> dataForOptimisation:
    """
    Build dataForOptimisation for the full historical window.

    Includes:
        - common date index
        - covariance matrices across time
        - deflated trading costs
        - unrounded target positions
        - FX rates and multipliers
        - capital and prices

    Returns
    -------
    dataForOptimisation
    """
    common_index = get_common_index(unrounded_position_contracts_dict)

    list_of_covariance_matrices = calculate_covariance_matrices(
        adjusted_prices_dict=adjusted_prices_dict,
        current_prices_dict=current_prices_dict,
        std_dev_dict=std_dev_dict,
    )

    deflated_costs_dict = calculate_deflated_costs_dict(
        std_dev_dict=std_dev_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        fx_series_dict=fx_series_dict,
    )

    return dataForOptimisation(
        capital=capital,
        deflated_costs_dict=deflated_costs_dict,
        fx_series_dict=fx_series_dict,
        list_of_covariance_matrices=list_of_covariance_matrices,
        multipliers=multipliers,
        common_index=common_index,
        unrounded_position_contracts_dict=unrounded_position_contracts_dict,
        current_prices_dict=current_prices_dict,
    )

def calculate_deflated_costs_dict(
    cost_per_contract_dict: dict,
    std_dev_dict: dict,
    fx_series_dict: dict,
) -> dict:
    """
    Compute deflated trading costs (scaled by volatility) and convert them
    to base currency using FX series.

    Steps:
        1. For each instrument, compute local deflated costs using volatility.
        2. Align FX series to the deflated-cost index.
        3. Convert costs to base currency.

    Parameters
    ----------
    cost_per_contract_dict : dict
        Mapping {instrument: cost_per_contract}.
    std_dev_dict : dict
        Mapping {instrument: standardDeviation}, providing volatility series.
    fx_series_dict : dict
        Mapping {instrument: FX rate series} for conversion to base currency.

    Returns
    -------
    dict
        Mapping {instrument: pd.Series} with deflated and FX-adjusted costs.
    """
    deflated_costs_dict = {
        instrument_code: calculated_deflated_costs_base_currency(
            stddev_series=std_dev_dict[instrument_code],
            cost_per_contract=cost_per_contract_dict[instrument_code],
            fx_series=fx_series_dict[instrument_code],
        )
        for instrument_code in cost_per_contract_dict.keys()
    }

    return deflated_costs_dict


def calculated_deflated_costs_base_currency(
    stddev_series: standardDeviation,
    cost_per_contract: float,
    fx_series: pd.Series
) -> pd.Series:
    """
    Convert local deflated costs into base currency.

    Parameters
    ----------
    stddev_series : standardDeviation
        Volatility estimator providing variance scale.
    cost_per_contract : float
        Cost per contract in local currency.
    fx_series : pd.Series
        FX rate series for conversion to base currency.

    Returns
    -------
    pd.Series
        Deflated cost series in base currency.
    """
    deflated_costs_local = calculate_deflated_costs(
        stddev_series=stddev_series,
        cost_per_contract=cost_per_contract,
    )

    fx_series_aligned = fx_series.reindex(deflated_costs_local.index).ffill()
    deflated_costs_base = deflated_costs_local * fx_series_aligned

    return deflated_costs_base


def get_data_for_relevant_date(
    relevant_date: datetime,
    data_for_optimisation: dataForOptimisation
) -> dataForSinglePeriod:
    """
    Build the optimisation data structure for a specific date.

    Extracts:
        - unrounded positions at the date
        - current prices
        - FX rates
        - covariance matrix (most recent before date)
        - deflated costs

    Parameters
    ----------
    relevant_date : datetime
        Date for which to extract optimisation inputs.
    data_for_optimisation : dataForOptimisation
        Master container holding historical data.

    Returns
    -------
    dataForSinglePeriod
        All required optimisation input for the given date.
    """
    unrounded_optimal_positions = positionContracts(
        get_values_for_date_as_dict(
            relevant_date,
            dict_with_values=data_for_optimisation.unrounded_position_contracts_dict,
        )
    )

    current_prices_this_period = get_values_for_date_as_dict(
        relevant_date,
        dict_with_values=data_for_optimisation.current_prices_dict,
    )

    fx_rates_this_period = get_values_for_date_as_dict(
        relevant_date,
        dict_with_values=data_for_optimisation.fx_series_dict,
    )

    covariance_this_period = (
        data_for_optimisation.list_of_covariance_matrices
        .most_recent_covariance_before_date(relevant_date)
    )

    current_cost_per_contract = get_values_for_date_as_dict(
        relevant_date=relevant_date,
        dict_with_values=data_for_optimisation.deflated_costs_dict,
    )

    return dataForSinglePeriod(
        capital=data_for_optimisation.capital,
        covariance_this_period=covariance_this_period,
        current_cost_per_contract=current_cost_per_contract,
        fx_rates_this_period=fx_rates_this_period,
        multipliers=data_for_optimisation.multipliers,
        current_prices_this_period=current_prices_this_period,
        unrounded_optimal_positions=unrounded_optimal_positions,
    )


def which_assets_have_data(data_for_single_period: dataForSinglePeriod) -> list:
    """
    Determine which assets have complete data (covariance, price, FX, and cost).

    Parameters
    ----------
    data_for_single_period : dataForSinglePeriod
        Period-specific data object.

    Returns
    -------
    list
        Set of assets with valid data across all required fields.
    """
    has_cov = data_for_single_period.covariance_this_period.assets_with_data()
    has_price = keys_with_data_in_dict(data_for_single_period.current_prices_this_period)
    has_fx = keys_with_data_in_dict(data_for_single_period.fx_rates_this_period)
    has_costs = keys_with_data_in_dict(data_for_single_period.current_cost_per_contract)

    return list(set(has_cov) & set(has_price) & set(has_fx) & set(has_costs))


def which_assets_without_data(
    data_for_single_period: dataForSinglePeriod,
    assets_with_data: list
) -> list:
    """
    Return the list of assets missing required data.

    Parameters
    ----------
    data_for_single_period : dataForSinglePeriod
        Data container for the date.
    assets_with_data : list
        Assets that DO have valid data.

    Returns
    -------
    list
        Assets missing one or more required fields.
    """
    assets = data_for_single_period.covariance_this_period.columns
    return list(set(assets) - set(assets_with_data))


def keys_with_data_in_dict(some_dict: dict):
    """
    Return keys where the corresponding value is not NaN.

    Parameters
    ----------
    some_dict : dict
        Mapping {key: numeric_value}.

    Returns
    -------
    list
        Keys with valid numeric values.
    """
    return [key for key, value in some_dict.items() if not np.isnan(value)]


def data_for_single_period_with_valid_assets_only(
    data_for_single_period: dataForSinglePeriod,
    assets_with_data: list
) -> dataForSinglePeriod:
    """
    Restrict the dataForSinglePeriod to contain only assets with valid data.

    Ensures:
        - covariance subset
        - filtered unrounded positions
        - filtered FX, prices, costs, multipliers

    Parameters
    ----------
    data_for_single_period : dataForSinglePeriod
        Full data structure.
    assets_with_data : list
        Assets that have data available.

    Returns
    -------
    dataForSinglePeriod
        Cleaned object containing only valid assets.
    """
    data_for_single_period.covariance_this_period = (
        data_for_single_period.covariance_this_period.subset(assets_with_data)
    )

    data_for_single_period.unrounded_optimal_positions = (
        data_for_single_period.unrounded_optimal_positions
        .with_selected_assets_only(assets_with_data)
    )

    data_for_single_period.fx_rates_this_period = dict_with_selected_assets_only(
        data_for_single_period.fx_rates_this_period, assets_with_data
    )

    data_for_single_period.current_prices_this_period = dict_with_selected_assets_only(
        data_for_single_period.current_prices_this_period, assets_with_data
    )

    data_for_single_period.current_cost_per_contract = dict_with_selected_assets_only(
        data_for_single_period.current_cost_per_contract, assets_with_data
    )

    data_for_single_period.multipliers = dict_with_selected_assets_only(
        data_for_single_period.multipliers, assets_with_data
    )

    return data_for_single_period


def get_initial_positions(position_contracts_dict: dict) -> positionContracts:
    """
    Generate the initial position vector (all zeros), used before optimisation starts.

    Parameters
    ----------
    position_contracts_dict : dict
        Mapping {instrument: initial_unrounded_contracts}.

    Returns
    -------
    positionContracts
        Position object with all zeros for all assets.
    """
    instrument_list = list(position_contracts_dict.keys())
    return positionContracts.allzeros(instrument_list)


def from_df_to_dict_of_series(position_df: pd.DataFrame) -> dict:
    """
    Convert a DataFrame of positions into a dict of Series.

    Parameters
    ----------
    position_df : pd.DataFrame
        DataFrame where columns are instruments and rows are dates.

    Returns
    -------
    dict
        Mapping {instrument: pd.Series} aligned with the DataFrame's index.
    """
    asset_names = list(position_df.columns)
    return {key: position_df[key] for key in asset_names}
