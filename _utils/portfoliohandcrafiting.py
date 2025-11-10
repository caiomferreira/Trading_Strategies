from scipy.cluster import hierarchy as sch
import pandas as pd
import numpy as np

from _utils.core_functions import *

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

#%% Portfolio Creation - Basics

def calculate_position_series_given_variable_risk_for_dict(
    capital: float,
    risk_target_tau: float,
    idm: float,
    weights: dict,
    fx_series_dict: dict,
    multipliers: dict,
    std_dev_dict: dict,
) -> dict:
    """
    Compute a dictionary of position time series (contracts held) for multiple instruments,
    accounting for variable risk and instrument-specific characteristics.

    Parameters
    ----------
    capital : float
        Total portfolio capital available for allocation.
    risk_target_tau : float
        Target risk per time period (e.g., daily or weekly volatility target).
    idm : float
        Diversification multiplier (Inverse of portfolio correlation-adjusted risk).
    weights : dict
        Portfolio weights per instrument, where keys are instrument codes.
    fx_series_dict : dict
        Dictionary of FX rate time series per instrument.
    multipliers : dict
        Contract multipliers per instrument.
    std_dev_dict : dict
        Rolling standard deviation series (risk estimates) per instrument.

    Returns
    -------
    dict
        Dictionary mapping each instrument code to its corresponding position size time series
        (in number of contracts).
    """
    position_series_dict = {
        instrument_code: calculate_position_series_given_variable_risk(
            capital=capital * idm * weights[instrument_code],
            risk_target_tau=risk_target_tau,
            multiplier=multipliers[instrument_code],
            fx=fx_series_dict[instrument_code],
            instrument_risk=std_dev_dict[instrument_code],
        )
        for instrument_code in std_dev_dict.keys()
    }
    return position_series_dict

# portfolio returns

def calculate_perc_returns_for_dict(
    position_contracts_dict: dict,
    adjusted_prices: dict,
    multipliers: dict,
    fx_series: dict,
    capital: float,
) -> dict:
    """
    Compute percentage return series for multiple instruments, 
    given position sizes and adjusted prices.

    Parameters
    ----------
    position_contracts_dict : dict
        Dictionary of position size series (contracts held) per instrument.
    adjusted_prices : dict
        Dictionary of adjusted price series per instrument.
    multipliers : dict
        Dictionary of contract multipliers per instrument.
    fx_series : dict
        Dictionary of FX rate series per instrument.
    capital : float
        Portfolio capital used as the base for percentage return computation.

    Returns
    -------
    dict
        Dictionary mapping each instrument code to its percentage return time series.
    """
    perc_returns_dict = {
        instrument_code: calculate_perc_returns(
            position_contracts_held=position_contracts_dict[instrument_code],
            adjusted_price=adjusted_prices[instrument_code],
            multiplier=multipliers[instrument_code],
            fx_series=fx_series[instrument_code],
            capital_required=capital,
        )
        for instrument_code in position_contracts_dict.keys()
    }
    return perc_returns_dict

# portfolio returns with costs
def calculate_perc_returns_for_dict_with_costs(
    position_contracts_dict: dict,
    adjusted_prices: dict,
    multipliers: dict,
    fx_series: dict,
    capital: float,
    cost_per_contract_dict: dict,
    std_dev_dict: dict,
) -> dict:

    perc_returns_dict = dict(
        [
            (
                instrument_code,
                calculate_perc_returns_with_costs(
                    position_contracts_held=position_contracts_dict[instrument_code],
                    adjusted_price=adjusted_prices[instrument_code],
                    multiplier=multipliers[instrument_code],
                    fx_series=fx_series[instrument_code],
                    capital_required=capital,
                    cost_per_contract=cost_per_contract_dict[instrument_code],
                    stdev_series=std_dev_dict[instrument_code],
                ),
            )
            for instrument_code in position_contracts_dict.keys()
        ]
    )

    return perc_returns_dict

def aggregate_returns(perc_returns_dict: dict) -> pd.Series:
    """
    Aggregate percentage return series from multiple instruments into a single portfolio return series.

    Parameters
    ----------
    perc_returns_dict : dict
        Dictionary mapping instrument codes to their respective percentage return series.

    Returns
    -------
    pd.Series
        Aggregated (summed) portfolio return series across all instruments.
    """
    both_returns = perc_returns_to_df(perc_returns_dict)
    return both_returns.sum(axis=1)


def perc_returns_to_df(perc_returns_dict: dict) -> pd.DataFrame:
    """
    Convert a dictionary of percentage return series into a single DataFrame.

    Parameters
    ----------
    perc_returns_dict : dict
        Dictionary mapping instrument codes to return series.

    Returns
    -------
    pd.DataFrame
        DataFrame where columns correspond to instrument return series and 
        index aligns on common dates (rows with all NaN values are dropped).
    """
    both_returns = pd.concat(perc_returns_dict, axis=1)
    both_returns = both_returns.dropna(how="all")
    return both_returns


def minimum_capital_for_sub_strategy(
    multiplier: float,
    price: float,
    fx: float,
    instrument_risk_ann_perc: float,
    risk_target: float,
    idm: float,
    weight: float,
    contracts: int = 4,
) -> float:
    """
    Calculate the minimum capital required to run a sub-strategy (instrument position)
    given risk, price, and diversification parameters.

    Formula
    --------
    Minimum capital = (contracts x multiplier x price x FX x σ_i) / (risk_target x IDM x weight)

    Parameters
    ----------
    multiplier : float
        Contract multiplier (e.g., contract size in local currency).
    price : float
        Current instrument price.
    fx : float
        FX rate relative to portfolio base currency.
    instrument_risk_ann_perc : float
        Annualized volatility of the instrument (as a percentage, e.g., 0.20 = 20%).
    risk_target : float
        Portfolio annualized target risk.
    idm : float
        Diversification multiplier (accounts for portfolio correlation).
    weight : float
        Portfolio weight assigned to the instrument.
    contracts : int, default=4
        Default number of contracts assumed for minimum position sizing.

    Returns
    -------
    float
        Minimum required capital for the instrument, in portfolio base currency.
    """
    return (
        contracts
        * multiplier
        * price
        * fx
        * instrument_risk_ann_perc
        / (risk_target * idm * weight)
    )

#%% =======================================
# Portfolio Handcraftiin - Cluster Creations

class correlationEstimate(object):
    """Encapsulates a correlation matrix and provides helper methods."""

    def __init__(self, values: pd.DataFrame):
        """Initialize with a pandas DataFrame (square correlation matrix)."""
        columns = values.columns
        values = values.values
        self._values = values
        self._columns = columns

    def __repr__(self):
        """Return a string representation as a pandas DataFrame."""
        return str(self.as_pd())

    def __len__(self):
        """Return the number of assets."""
        return len(self.columns)

    def as_pd(self) -> pd.DataFrame:
        """Return the correlation matrix as a pandas DataFrame."""
        values = self.values
        columns = self.columns
        return pd.DataFrame(values, index=columns, columns=columns)

    @property
    def values(self) -> np.array:
        """Return the correlation matrix as a numpy array."""
        return self._values

    @property
    def columns(self) -> list:
        """Return the list of asset names."""
        return self._columns

    @property
    def size(self) -> int:
        """Return the number of assets in the correlation matrix."""
        return len(self.columns)

    def subset(self, subset_of_asset_names: list):
        """Return a new correlationEstimate for a subset of assets."""
        as_pd = self.as_pd()
        subset_pd = as_pd.loc[subset_of_asset_names, subset_of_asset_names]
        new_correlation = correlationEstimate(subset_pd)
        return new_correlation


def cluster_correlation_matrix(corr_matrix: correlationEstimate, cluster_size: int = 2, print_trace: bool = True):
    """Perform hierarchical clustering on a correlation matrix.

    Args:
        corr_matrix: correlationEstimate instance.
        cluster_size: number of desired clusters.
        print_trace: whether to print cluster information.

    Returns:
        List of clusters as lists of asset names.
    """
    clusters = get_list_of_clusters_for_correlation_matrix(corr_matrix, cluster_size=cluster_size)
    clusters_as_names = from_cluster_index_to_asset_names(clusters, corr_matrix)
    if print_trace:
        print("Cluster split:", clusters_as_names)
    return clusters_as_names


def get_list_of_clusters_for_correlation_matrix(corr_matrix: np.array, cluster_size: int = 2) -> list:
    """Return a list of cluster indices for each asset."""
    corr_as_np = corr_matrix.values
    try:
        clusters = get_list_of_clusters_for_correlation_matrix_as_np(corr_as_np, cluster_size=cluster_size)
    except Exception:
        clusters = arbitrary_split_of_correlation_matrix(corr_as_np, cluster_size=cluster_size)
    return clusters


def get_list_of_clusters_for_correlation_matrix_as_np(corr_as_np: np.array, cluster_size: int = 2) -> list:
    """Compute hierarchical linkage and return cluster assignments."""
    d = sch.distance.pdist(corr_as_np)
    L = sch.linkage(d, method="complete")
    cutoff = cutoff_distance_to_guarantee_N_clusters(corr_as_np, L=L, cluster_size=cluster_size)
    ind = sch.fcluster(L, cutoff, "distance")
    ind = list(ind)
    if max(ind) > cluster_size:
        raise Exception(f"Couldn't cluster into {cluster_size} clusters")
    return ind


def cutoff_distance_to_guarantee_N_clusters(corr_as_np: np.array, L: np.array, cluster_size: int = 2):
    """Compute cutoff distance to ensure a specific number of clusters."""
    N = len(corr_as_np)
    return L[N - cluster_size][2] - 0.000001


def arbitrary_split_of_correlation_matrix(corr_matrix: np.array, cluster_size: int = 2) -> list:
    """Fallback split: assign assets alternately to clusters."""
    count_assets = len(corr_matrix)
    return arbitrary_split_for_asset_length(count_assets, cluster_size=cluster_size)


def arbitrary_split_for_asset_length(count_assets: int, cluster_size: int = 2) -> list:
    """Assign assets cyclically to clusters."""
    return [(x % cluster_size) + 1 for x in range(count_assets)]


def from_cluster_index_to_asset_names(clusters: list, corr_matrix: correlationEstimate) -> list:
    """Convert cluster indices into lists of asset names."""
    all_clusters = list(set(clusters))
    asset_names = corr_matrix.columns
    list_of_asset_clusters = [
        get_asset_names_for_cluster_index(cluster_id, clusters, asset_names)
        for cluster_id in all_clusters
    ]
    return list_of_asset_clusters


def get_asset_names_for_cluster_index(cluster_id: int, clusters: list, asset_names: list):
    """Return list of asset names that belong to a specific cluster."""
    list_of_assets = [
        asset for asset, cluster in zip(asset_names, clusters) if cluster == cluster_id
    ]
    return list_of_assets


class portfolioWeights(dict):
    """Dictionary subclass representing portfolio weights."""

    @property
    def weights(self):
        """Return list of weight values."""
        return list(self.values())

    @property
    def assets(self):
        """Return list of asset names."""
        return list(self.keys())

    def multiply_by_float(self, multiplier: float):
        """Scale all weights by a scalar multiplier."""
        list_of_assets = self.assets
        list_of_weights = [self[asset] for asset in list_of_assets]
        list_of_weights_multiplied = [weight * multiplier for weight in list_of_weights]
        return portfolioWeights.from_weights_and_keys(
            list_of_weights=list_of_weights_multiplied,
            list_of_keys=list_of_assets
        )

    @classmethod
    def from_list_of_subportfolios(portfolioWeights, list_of_portfolio_weights):
        """Combine multiple portfolioWeights objects into one aggregated portfolio."""
        list_of_unique_asset_names = list(
            set(
                flatten_list(
                    [subportfolio.assets for subportfolio in list_of_portfolio_weights]
                )
            )
        )
        portfolio_weights = portfolioWeights.allzeros(list_of_unique_asset_names)
        for subportfolio_weights in list_of_portfolio_weights:
            for asset_name in subportfolio_weights.assets:
                portfolio_weights[asset_name] = (
                    portfolio_weights[asset_name] + subportfolio_weights[asset_name]
                )
        return portfolio_weights

    @classmethod
    def allzeros(portfolioWeights, list_of_keys: list):
        """Create portfolio with all weights equal to zero."""
        return portfolioWeights.all_one_value(list_of_keys, value=0.0)

    @classmethod
    def all_one_value(portfolioWeights, list_of_keys: list, value=0.0):
        """Create portfolio with the same value for all assets."""
        return portfolioWeights.from_weights_and_keys(
            list_of_weights=[value] * len(list_of_keys), list_of_keys=list_of_keys
        )

    @classmethod
    def from_weights_and_keys(portfolioWeights, list_of_weights: list, list_of_keys: list):
        """Create a portfolioWeights object from lists of weights and asset names."""
        assert len(list_of_keys) == len(list_of_weights)
        pweights_as_list = [
            (key, weight) for key, weight in zip(list_of_keys, list_of_weights)
        ]
        return portfolioWeights(pweights_as_list)


def flatten_list(some_list):
    """Flatten a nested list."""
    flattened = [item for sublist in some_list for item in sublist]
    return flattened


def one_over_n_weights_given_asset_names(list_of_asset_names: list) -> portfolioWeights:
    """Create equal-weight portfolio given a list of asset names."""
    weight = 1.0 / len(list_of_asset_names)
    return portfolioWeights([(asset_name, weight) for asset_name in list_of_asset_names])


class handcraftPortfolio(object):
    """Portfolio constructed from a correlation matrix using hierarchical clustering."""

    def __init__(self, correlation: correlationEstimate, print_trace: bool = True):
        """Initialize with a correlationEstimate object.

        Args:
            correlation: correlationEstimate object.
            print_trace: whether to print cluster split information.
        """
        self._correlation = correlation
        self._print_trace = print_trace

    @property
    def correlation(self) -> correlationEstimate:
        """Return the correlationEstimate object."""
        return self._correlation

    @property
    def size(self) -> int:
        """Return the number of assets in the portfolio."""
        return len(self.correlation)

    @property
    def asset_names(self) -> list:
        """Return the list of asset names."""
        return list(self.correlation.columns)

    def weights(self) -> portfolioWeights:
        """Compute portfolio weights using clustering logic."""
        if self.size <= 2:
            raw_weights = self.risk_weights_this_portfolio()
        else:
            raw_weights = self.aggregated_risk_weights()
        return raw_weights

    def risk_weights_this_portfolio(self) -> portfolioWeights:
        """Assign equal weights (1/N) to all assets."""
        asset_names = self.asset_names
        raw_weights = one_over_n_weights_given_asset_names(asset_names)
        return raw_weights

    def aggregated_risk_weights(self):
        """Aggregate risk weights from clustered sub-portfolios."""
        sub_portfolios = create_sub_portfolios_from_portfolio(self, print_trace=self._print_trace)
        aggregate_risk_weights = aggregate_risk_weights_over_sub_portfolios(sub_portfolios)
        return aggregate_risk_weights

    def subset(self, subset_of_asset_names: list):
        """Return a new handcraftPortfolio containing a subset of assets."""
        return handcraftPortfolio(self.correlation.subset(subset_of_asset_names), print_trace=self._print_trace) 
    
    def get_hierarchy_tree(self):
        """Return full hierarchical tree of the portfolio as nested dict."""
        return self._build_tree_recursive(self.correlation)

    def _build_tree_recursive(self, corr_obj: correlationEstimate):
        """Recursively build nested tree representing clustering hierarchy."""
        clusters = cluster_correlation_matrix(corr_obj, 2, print_trace=self._print_trace)
        tree = []
        for subcluster in clusters:
            sub_corr = corr_obj.subset(subcluster)
            if sub_corr.size <= 2:
                tree.append(subcluster)
            else:
                subtree = self._build_tree_recursive(sub_corr)
                tree.append(subtree)
        return tree


def create_sub_portfolios_from_portfolio(handcraft_portfolio: handcraftPortfolio, print_trace: bool = True):
    """Create sub-portfolios by clustering the main portfolio."""
    clusters_as_names = cluster_correlation_matrix(
        handcraft_portfolio.correlation,
        print_trace=print_trace
    )
    sub_portfolios = create_sub_portfolios_given_clusters(
        clusters_as_names, handcraft_portfolio
    )
    return sub_portfolios


def create_sub_portfolios_given_clusters(clusters_as_names: list, handcraft_portfolio: handcraftPortfolio) -> list:
    """Generate sub-portfolios for each cluster."""
    list_of_sub_portfolios = [
        handcraft_portfolio.subset(subset_of_asset_names)
        for subset_of_asset_names in clusters_as_names
    ]
    return list_of_sub_portfolios


def aggregate_risk_weights_over_sub_portfolios(sub_portfolios: list) -> portfolioWeights:
    """Aggregate portfolio weights across all sub-portfolios equally."""
    asset_count = len(sub_portfolios)
    weights_for_each_subportfolio = [1.0 / asset_count] * asset_count
    risk_weights_by_portfolio = [sub_portfolio.weights() for sub_portfolio in sub_portfolios]
    multiplied_risk_weights_by_portfolio = [
        sub_portfolio_weights.multiply_by_float(weight_for_subportfolio)
        for weight_for_subportfolio, sub_portfolio_weights in zip(
            weights_for_each_subportfolio, risk_weights_by_portfolio
        )
    ]
    aggregate_weights = portfolioWeights.from_list_of_subportfolios(multiplied_risk_weights_by_portfolio)
    return aggregate_weights


#%% ======================================
# Portfolio Instrument Selection

import pandas as pd
import numpy as np


def select_first_static_instrument(instrument_config: pd.DataFrame,
                                   approx_number_of_instruments: int,
                                   approx_IDM: float,
                                   capital: float,
                                   risk_target: float,
                                   position_turnover: float) -> str:
    """
    Select the first instrument in a static portfolio based on risk-adjusted cost 
    and minimum capital constraints.

    Parameters
    ----------
    instrument_config : pd.DataFrame
        Instrument configuration table with columns such as 
        'fx_rate', 'ann_std', 'price', 'multiplier', 'SR_cost', and 'rolls_per_year'.
    approx_number_of_instruments : int
        Approximate number of instruments expected in the portfolio.
    approx_IDM : float
        Approximate diversification multiplier.
    capital : float
        Available trading capital.
    risk_target : float
        Annualized target risk (e.g., 0.15 = 15%).
    position_turnover : float
        Expected average portfolio turnover (in trades per year).

    Returns
    -------
    str
        Code of the cheapest instrument adjusted for risk that satisfies 
        the minimum capital requirement.
    """
    approx_initial_weight = 1.0 / approx_number_of_instruments
    instrument_list = list(instrument_config.index)
    instruments_okay_for_minimum_capital = [
        instrument_code
        for instrument_code in instrument_list
        if minimum_capital_okay_for_instrument(
            instrument_code=instrument_code,
            instrument_config=instrument_config,
            capital=capital,
            weight=approx_initial_weight,
            idm=approx_IDM,
            risk_target=risk_target
        )
    ]

    cheapest_instrument = lowest_risk_adjusted_cost_given_instrument_list(
        instruments_okay_for_minimum_capital,
        instrument_config=instrument_config,
        position_turnover=position_turnover
    )
    return cheapest_instrument


def minimum_capital_okay_for_instrument(instrument_code: str,
                                         instrument_config: pd.DataFrame,
                                         idm: float,
                                         weight: float,
                                         risk_target: float,
                                         capital: float) -> bool:
    """
    Check whether the required minimum capital for a given instrument 
    fits within the available portfolio capital.

    Parameters
    ----------
    instrument_code : str
        Instrument identifier.
    instrument_config : pd.DataFrame
        DataFrame containing configuration parameters for each instrument.
    idm : float
        Diversification multiplier.
    weight : float
        Target portfolio weight of the instrument.
    risk_target : float
        Portfolio target annualized risk.
    capital : float
        Available capital.

    Returns
    -------
    bool
        True if the required minimum capital is less than or equal to available capital.
    """
    config_for_instrument = instrument_config.loc[instrument_code]
    minimum_capital = minimum_capital_for_sub_strategy(
        fx=config_for_instrument.fx_rate,
        idm=idm,
        weight=weight,
        instrument_risk_ann_perc=config_for_instrument.ann_std,
        price=config_for_instrument.price,
        multiplier=config_for_instrument.multiplier,
        risk_target=risk_target
    )
    return minimum_capital <= capital


def lowest_risk_adjusted_cost_given_instrument_list(
        instrument_list: list,
        instrument_config: pd.DataFrame,
        position_turnover: float) -> str:
    """
    Return the instrument with the lowest risk-adjusted cost 
    from a list of eligible instruments.

    Parameters
    ----------
    instrument_list : list
        List of eligible instrument codes.
    instrument_config : pd.DataFrame
        Instrument configuration table.
    position_turnover : float
        Expected portfolio turnover per year.

    Returns
    -------
    str
        Instrument code with the lowest risk-adjusted cost.
    """
    list_of_risk_adjusted_cost_by_instrument = [
        risk_adjusted_cost_for_instrument(instrument_code,
                                          instrument_config=instrument_config,
                                          position_turnover=position_turnover)
        for instrument_code in instrument_list
    ]
    index_min = get_min_index(list_of_risk_adjusted_cost_by_instrument)
    return instrument_list[index_min]


def get_min_index(x: list) -> int:
    """Return the index of the minimum value in a list."""
    return get_func_index(x, min)


def get_max_index(x: list) -> int:
    """Return the index of the maximum value in a list."""
    return get_func_index(x, max)


def get_func_index(x: list, func) -> int:
    """
    Return the index corresponding to a function applied to list values 
    (e.g., min or max).

    Parameters
    ----------
    x : list
        List of numeric values.
    func : callable
        Aggregation function such as min or max.

    Returns
    -------
    int
        Index of the element resulting from applying the function.
    """
    return func(range(len(x)), key=x.__getitem__)


def risk_adjusted_cost_for_instrument(instrument_code: str,
                                      instrument_config: pd.DataFrame,
                                      position_turnover: float) -> float:
    """
    Compute the risk-adjusted trading cost for an instrument 
    expressed in Sharpe Ratio (SR) units.

    Parameters
    ----------
    instrument_code : str
        Instrument identifier.
    instrument_config : pd.DataFrame
        Instrument configuration table containing SR_cost and rolls_per_year.
    position_turnover : float
        Expected number of rebalancing events per year.

    Returns
    -------
    float
        Risk-adjusted cost in SR units.
    """
    cfg = instrument_config.loc[instrument_code]
    return cfg.SR_cost * (cfg.rolls_per_year + position_turnover)


def calculate_SR_for_selected_instruments(selected_instruments: list,
                                          pre_cost_SR: float,
                                          instrument_config: pd.DataFrame,
                                          position_turnover: float,
                                          correlation_matrix,
                                          capital: float,
                                          risk_target: float) -> float:
    """
    Calculate the expected Sharpe Ratio for a given selection of instruments.

    Parameters
    ----------
    selected_instruments : list
        List of selected instrument codes.
    pre_cost_SR : float
        Expected Sharpe Ratio before transaction costs.
    instrument_config : pd.DataFrame
        Instrument configuration table.
    position_turnover : float
        Expected turnover per year.
    correlation_matrix : correlationEstimate
        Correlation matrix between instruments.
    capital : float
        Available trading capital.
    risk_target : float
        Target portfolio risk (annualized).

    Returns
    -------
    float
        Expected post-cost Sharpe Ratio of the portfolio, 
        or a large negative number if capital constraints are violated.
    """
    weights = calculate_portfolio_weights(selected_instruments, correlation_matrix)
    if not check_minimum_capital_ok(weights, correlation_matrix, risk_target, instrument_config, capital):
        return -999999999999
    return calculate_SR_of_portfolio(weights, pre_cost_SR, instrument_config, position_turnover, correlation_matrix)


def calculate_portfolio_weights(selected_instruments: list, correlation_matrix) -> "portfolioWeights":
    """
    Compute portfolio weights for selected instruments 
    using the correlation structure.

    Parameters
    ----------
    selected_instruments : list
        List of instrument codes.
    correlation_matrix : correlationEstimate
        Correlation matrix among instruments.

    Returns
    -------
    portfolioWeights
        Object representing normalized portfolio weights.
    """
    if len(selected_instruments) == 1:
        return portfolioWeights.from_weights_and_keys([1.0], selected_instruments)
    subset_matrix = correlation_matrix.subset(selected_instruments)
    return handcraftPortfolio(subset_matrix).weights()


def check_minimum_capital_ok(portfolio_weights,
                             correlation_matrix,
                             risk_target: float,
                             instrument_config: pd.DataFrame,
                             capital: float) -> bool:
    """
    Verify whether all instruments in the portfolio meet 
    their respective minimum capital requirements.

    Returns
    -------
    bool
        True if all instruments meet capital constraints, otherwise False.
    """
    idm = calculate_idm(portfolio_weights, correlation_matrix)
    for instrument_code in portfolio_weights.assets:
        weight = portfolio_weights[instrument_code]
        if not minimum_capital_okay_for_instrument(instrument_code, instrument_config,
                                                   capital=capital,
                                                   risk_target=risk_target,
                                                   idm=idm,
                                                   weight=weight):
            return False
    return True


def calculate_idm(portfolio_weights, correlation_matrix) -> float:
    """
    Compute the diversification multiplier (IDM) 
    based on portfolio weights and correlation matrix.

    Returns
    -------
    float
        Diversification multiplier (higher values indicate better diversification).
    """
    if len(portfolio_weights.assets) == 1:
        return 1.0
    sub_corr = correlation_matrix.subset(portfolio_weights.assets)
    return div_multiplier_from_np(np.array(portfolio_weights.weights), sub_corr.values)


def div_multiplier_from_np(weights_np: np.array, corr_np: np.array) -> float:
    """
    Compute the diversification multiplier (IDM) from numpy arrays.

    Returns
    -------
    float
        IDM = 1 / sqrt(w.T * Corr * w)
    """
    variance = weights_np.dot(corr_np).dot(weights_np)
    return 1.0 / np.sqrt(variance)


def calculate_SR_of_portfolio(portfolio_weights,
                              pre_cost_SR: float,
                              instrument_config: pd.DataFrame,
                              position_turnover: float,
                              correlation_matrix) -> float:
    """
    Compute the expected portfolio Sharpe Ratio after transaction costs.

    Returns
    -------
    float
        Expected net Sharpe Ratio.
    """
    mean = calculate_expected_mean_for_portfolio(portfolio_weights, pre_cost_SR,
                                                 instrument_config, position_turnover)
    std = calculate_expected_std_for_portfolio(portfolio_weights, correlation_matrix)
    return mean / std


def calculate_expected_mean_for_portfolio(portfolio_weights,
                                          pre_cost_SR: float,
                                          instrument_config: pd.DataFrame,
                                          position_turnover: float) -> float:
    """Compute the weighted sum of expected post-cost Sharpe Ratios for all instruments."""
    return sum(
        calculate_expected_mean_for_instrument_in_portfolio(i, portfolio_weights,
                                                            pre_cost_SR, instrument_config,
                                                            position_turnover)
        for i in portfolio_weights.assets
    )


def calculate_expected_mean_for_instrument_in_portfolio(instrument_code: str,
                                                        portfolio_weights,
                                                        pre_cost_SR: float,
                                                        instrument_config: pd.DataFrame,
                                                        position_turnover: float) -> float:
    """Compute the contribution of a single instrument to portfolio mean return (in SR units)."""
    weight = portfolio_weights[instrument_code]
    SR_costs = risk_adjusted_cost_for_instrument(instrument_code, instrument_config, position_turnover)
    return weight * (pre_cost_SR - SR_costs)


def calculate_expected_std_for_portfolio(portfolio_weights, correlation_matrix) -> float:
    """Compute the expected portfolio standard deviation using the correlation matrix."""
    subset = correlation_matrix.subset(portfolio_weights.assets)
    return np.sqrt(variance_for_numpy(np.array(portfolio_weights.weights), subset.values))


def variance_for_numpy(weights: np.array, sigma: np.array) -> float:
    """Compute the portfolio variance given weights and a covariance/correlation matrix."""
    return weights.dot(sigma).dot(weights.T)


def choose_next_instrument(selected_instruments: list,
                           pre_cost_SR: float,
                           capital: float,
                           risk_target: float,
                           instrument_config: pd.DataFrame,
                           position_turnover: float,
                           correlation_matrix) -> str:
    """
    Select the next instrument to add to the portfolio 
    that maximizes the post-cost Sharpe Ratio.

    Returns
    -------
    str
        Code of the instrument that maximizes incremental Sharpe Ratio.
    """
    remaining_instruments = get_remaining_instruments(selected_instruments, instrument_config)
    SR_by_instrument = [
        calculate_SR_for_selected_instruments(selected_instruments + [i],
                                              correlation_matrix=correlation_matrix,
                                              capital=capital,
                                              pre_cost_SR=pre_cost_SR,
                                              instrument_config=instrument_config,
                                              risk_target=risk_target,
                                              position_turnover=position_turnover)
        for i in remaining_instruments
    ]
    return remaining_instruments[get_max_index(SR_by_instrument)]


def get_remaining_instruments(selected_instruments: list,
                              instrument_config: pd.DataFrame) -> list:
    """Return instruments not yet included in the portfolio."""
    return list(set(instrument_config.index) - set(selected_instruments))

#%% ====================================
# Portfolio Backtesting


def calculate_variable_standard_deviation_for_risk_targeting_from_dict(
    adjusted_prices: dict,
    current_prices: dict,
    use_perc_returns: bool = True,
    annualise_stdev: bool = True,
) -> dict:
    """
    Compute rolling or exponential standard deviation for risk targeting per instrument.

    Parameters
    ----------
    adjusted_prices : dict
        Dictionary of adjusted price Series for each instrument.
    current_prices : dict
        Dictionary of current prices for each instrument.
    use_perc_returns : bool, optional
        Whether to calculate percentage returns (default is True).
    annualise_stdev : bool, optional
        Whether to annualize the standard deviation (default is True).

    Returns
    -------
    dict
        Dictionary of standard deviation Series for each instrument.
    """
    std_dev_dict = dict(
        [
            (
                instrument_code,
                standardDeviation(
                    adjusted_price=adjusted_prices[instrument_code],
                    current_price=current_prices[instrument_code],
                    use_perc_returns=use_perc_returns,
                    annualise_stdev=annualise_stdev,
                ),
            )
            for instrument_code in adjusted_prices.keys()
        ]
    )
    return std_dev_dict


def calculate_position_series_given_variable_risk_for_dict(
    capital: float,
    risk_target_tau: float,
    idm: float,
    weights: dict,
    fx_series_dict: dict,
    multipliers: dict,
    std_dev_dict: dict,
) -> dict:
    """
    Calculate position sizes per instrument given variable risk and other parameters.

    Parameters
    ----------
    capital : float
        Total capital available for the strategy.
    risk_target_tau : float
        Target risk level (e.g., annualized volatility target).
    idm : float
        Instrument diversification multiplier.
    weights : dict
        Portfolio weights for each instrument.
    fx_series_dict : dict
        Dictionary of FX rate Series for each instrument.
    multipliers : dict
        Contract multipliers for each instrument.
    std_dev_dict : dict
        Standard deviation Series for each instrument.

    Returns
    -------
    dict
        Dictionary of position Series (number of contracts) per instrument.
    """
    position_series_dict = dict(
        [
            (
                instrument_code,
                calculate_position_series_given_variable_risk(
                    capital=capital * idm * weights[instrument_code],
                    risk_target_tau=risk_target_tau,
                    multiplier=multipliers[instrument_code],
                    fx=fx_series_dict[instrument_code],
                    instrument_risk=std_dev_dict[instrument_code],
                ),
            )
            for instrument_code in std_dev_dict.keys()
        ]
    )
    return position_series_dict


def calculate_perc_returns_for_dict(
    position_contracts_dict: dict,
    adjusted_prices: dict,
    multipliers: dict,
    fx_series: dict,
    capital: float,
) -> dict:
    """
    Calculate percentage return series for each instrument based on position and FX.

    Parameters
    ----------
    position_contracts_dict : dict
        Dictionary of position Series (contracts held) per instrument.
    adjusted_prices : dict
        Dictionary of adjusted price Series per instrument.
    multipliers : dict
        Contract multipliers per instrument.
    fx_series : dict
        Dictionary of FX rate Series per instrument.
    capital : float
        Total capital used to compute percentage returns.

    Returns
    -------
    dict
        Dictionary of percentage return Series per instrument.
    """
    perc_returns_dict = dict(
        [
            (
                instrument_code,
                calculate_perc_returns(
                    position_contracts_held=position_contracts_dict[instrument_code],
                    adjusted_price=adjusted_prices[instrument_code],
                    multiplier=multipliers[instrument_code],
                    fx_series=fx_series[instrument_code],
                    capital_required=capital,
                ),
            )
            for instrument_code in position_contracts_dict.keys()
        ]
    )
    return perc_returns_dict


def aggregate_returns(perc_returns_dict: dict) -> pd.Series:
    """
    Aggregate individual instrument returns into a single total portfolio return series.

    Parameters
    ----------
    perc_returns_dict : dict
        Dictionary of percentage return Series per instrument.

    Returns
    -------
    pd.Series
        Aggregated total portfolio return series.
    """
    both_returns = perc_returns_to_df(perc_returns_dict)
    agg = both_returns.sum(axis=1)
    agg.index = pd.to_datetime(agg.index)
    return agg


def perc_returns_to_df(perc_returns_dict: dict) -> pd.DataFrame:
    """
    Convert a dictionary of percentage returns into a DataFrame for analysis.

    Parameters
    ----------
    perc_returns_dict : dict
        Dictionary of percentage return Series per instrument.

    Returns
    -------
    pd.DataFrame
        DataFrame of percentage returns with instruments as columns.
    """
    both_returns = pd.concat(perc_returns_dict, axis=1)
    both_returns = both_returns.dropna(how="all")
    return both_returns


def minimum_capital_for_sub_strategy(
    multiplier: float,
    price: float,
    fx: float,
    instrument_risk_ann_perc: float,
    risk_target: float,
    idm: float,
    weight: float,
    contracts: int = 4,
):
    """
    Calculate the minimum required capital for a sub-strategy based on its parameters.

    Formula:
        (contracts x multiplier x price x FX x σ%) ÷ (risk_target x IDM x weight)

    Parameters
    ----------
    multiplier : float
        Contract multiplier for the instrument.
    price : float
        Current price of the instrument.
    fx : float
        FX rate relative to the portfolio's base currency.
    instrument_risk_ann_perc : float
        Annualized risk (standard deviation in percentage terms).
    risk_target : float
        Target annualized risk for the strategy.
    idm : float
        Instrument diversification multiplier.
    weight : float
        Weight assigned to the instrument within the strategy.
    contracts : int, optional
        Number of contracts used for the calculation (default is 4).

    Returns
    -------
    float
        Minimum capital required for the sub-strategy.
    """
    return (
        contracts
        * multiplier
        * price
        * fx
        * instrument_risk_ann_perc
        / (risk_target * idm * weight)
    )
