import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from _utils.core_functions import*

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



class stdevEstimate(dict):
    """
    Container for per-asset volatility (standard deviation) estimates.

    Behaves like a dict mapping:
        {asset: stdev_value_or_series}

    Provides helpers to:
        - return keys
        - filter assets with valid data
        - extract stdev values in a specific asset order
    """

    def list_of_keys(self) -> list:
        """Return list of asset names."""
        return list(self.keys())

    def assets_with_data(self) -> list:
        """
        Return assets whose stdev values are not NaN.

        Returns
        -------
        list
            Assets with valid (non-NaN) volatility values.
        """
        return [
            asset_name
            for asset_name, stdev in self.items()
            if not np.isnan(stdev)
        ]

    def list_in_key_order(self, list_of_assets) -> list:
        """
        Return stdev values ordered according to the provided list of assets.

        Parameters
        ----------
        list_of_assets : list
            Desired asset order.

        Returns
        -------
        list
            Stdev values in the requested order.
        """
        return [self[asset_name] for asset_name in list_of_assets]


class genericMatrixEstimate:
    """
    Generic matrix wrapper used for covariance and correlation matrices.

    Parameters
    ----------
    values : np.array or pd.DataFrame
        Matrix values.
    columns : list
        Asset names corresponding to matrix rows/columns.

    Provides:
        - DataFrame representation
        - subset operations
        - detection of assets with missing data
        - insertion of new assets with NaN rows/columns
    """

    def __init__(self, values: np.array, columns: list):
        if isinstance(values, pd.DataFrame):
            columns = values.columns
            values = values.values

        self._values = values
        self._columns = columns

    def __repr__(self):
        return str(self.as_df())

    def subset(self, subset_of_asset_names: list):
        """
        Extract a square submatrix using the given list of asset names.

        Returns
        -------
        genericMatrixEstimate
        """
        as_df = self.as_df()
        subset_df = as_df.loc[subset_of_asset_names, subset_of_asset_names]
        return self.from_pd(subset_df)

    def add_assets_with_nan_values(self, new_asset_names: list):
        """
        Expand matrix by adding new assets with full NaN rows/columns.

        Parameters
        ----------
        new_asset_names : list
            List of assets to append.

        Returns
        -------
        correlationEstimate
            New enlarged matrix with NaN blocks.
        """
        l1 = self.as_df()

        # right-block for existing rows
        r1 = pd.DataFrame(
            [[np.nan] * len(new_asset_names)] * len(self.columns),
            columns=new_asset_names,
            index=self.columns,
        )
        top_row = pd.concat([l1, r1], axis=1)

        # bottom-left block
        l2 = pd.DataFrame(
            [[np.nan] * len(self.columns)] * len(new_asset_names),
            columns=self.columns,
            index=new_asset_names,
        )
        # bottom-right block
        r2 = pd.DataFrame(
            [[np.nan] * len(new_asset_names)] * len(new_asset_names),
            columns=new_asset_names,
            index=new_asset_names,
        )
        bottom_row = pd.concat([l2, r2], axis=1)

        both_rows = pd.concat([top_row, bottom_row], axis=0)

        return correlationEstimate(values=both_rows.values, columns=both_rows.columns)

    @classmethod
    def from_pd(cls, matrix_as_pd: pd.DataFrame):
        """
        Build matrix estimate from a pandas DataFrame.

        Returns
        -------
        genericMatrixEstimate
        """
        return cls(matrix_as_pd.values, columns=list(matrix_as_pd.columns))

    def assets_with_data(self) -> list:
        """
        Return assets whose rows/columns have at least two non-NaN entries.

        Returns
        -------
        list
            Assets with usable covariance/correlation entries.
        """
        missing = self.assets_with_missing_data()
        return [c for c in self.columns if c not in missing]

    def assets_with_missing_data(self) -> list:
        """
        Detect assets whose data rows are too sparse for valid estimation.

        Returns
        -------
        list
            Assets considered missing (rows with <2 non-NaN).
        """
        na_row_count = (~self.as_df().isna()).sum() < 2
        return [key for key in na_row_count.keys() if na_row_count[key]]

    def as_df(self):
        """Return matrix as a pandas DataFrame."""
        return pd.DataFrame(self.values, columns=self.columns, index=self.columns)

    @property
    def size(self):
        """Return number of assets."""
        return len(self.columns)

    @property
    def values(self):
        """Return numpy array of matrix values."""
        return self._values

    @property
    def columns(self):
        """Return list of asset names."""
        return self._columns


class covarianceEstimate(genericMatrixEstimate):
    """
    Covariance matrix estimate.

    Inherits all functionality from genericMatrixEstimate.
    """
    pass


class correlationEstimate(genericMatrixEstimate):
    """
    Correlation matrix estimate with shrinkage utilities.

    Provides:
        - shrink_to_offdiag(): shrink correlations toward constant-off-diag prior
        - boring_corr_matrix(): create constant correlation matrix
        - shrink(): linear shrinkage toward prior
    """

    def shrink_to_offdiag(self, offdiag=0.0, shrinkage_corr: float = 1.0):
        """
        Shrink correlation matrix toward a constant off-diagonal prior.

        Parameters
        ----------
        offdiag : float
            Off-diagonal prior correlation.
        shrinkage_corr : float
            Shrinkage intensity (1.0 = full prior, 0.0 = raw).

        Returns
        -------
        correlationEstimate
        """
        prior_corr = self.boring_corr_matrix(offdiag=offdiag)
        return self.shrink(prior_corr=prior_corr, shrinkage_corr=shrinkage_corr)

    def boring_corr_matrix(self, offdiag: float = 0.99, diag: float = 1.0):
        """
        Create fully constant correlation matrix for same asset set.

        Returns
        -------
        correlationEstimate
        """
        return create_boring_corr_matrix(
            self.size, offdiag=offdiag, diag=diag, columns=self.columns
        )

    def shrink(self, prior_corr: "correlationEstimate", shrinkage_corr: float = 1.0):
        """
        Apply linear shrinkage between empirical and prior correlation matrix.

        Parameters
        ----------
        prior_corr : correlationEstimate
            Prior matrix.
        shrinkage_corr : float
            Amount of shrinkage.

        Returns
        -------
        correlationEstimate
        """
        if shrinkage_corr == 1.0:
            return prior_corr
        if shrinkage_corr == 0.0:
            return self

        shrunk_corr_values = (
            shrinkage_corr * prior_corr.values
            + (1 - shrinkage_corr) * self.values
        )
        return correlationEstimate(shrunk_corr_values, columns=self.columns)


def create_boring_corr_matrix(size: int, columns: list,
                              offdiag: float = 0.99, diag: float = 1.0) -> correlationEstimate:
    """
    Create constant correlation matrix.

    Returns
    -------
    correlationEstimate
    """
    corr_matrix_values = boring_corr_matrix_values(size, offdiag=offdiag, diag=diag)
    mtx = correlationEstimate(corr_matrix_values, columns=columns)
    mtx.is_boring = True
    return mtx


def boring_corr_matrix_values(size: int, offdiag: float = 0.99, diag: float = 1.0) -> np.array:
    """
    Generate numpy array for constant correlation matrix.

    Returns
    -------
    np.array
    """
    return np.array([
        [diag if i == j else offdiag for i in range(size)]
        for j in range(size)
    ])


@dataclass
class covarianceList:
    """
    Container for a list of covarianceEstimate objects and their corresponding dates.

    Provides:
        - most_recent_covariance_before_date(): find last available cov matrix.
    """
    cov_list: list
    fit_dates: list

    def most_recent_covariance_before_date(self, relevant_date: datetime) -> covarianceEstimate:
        """
        Fetch the covariance matrix computed most recently before a given date.

        Returns
        -------
        covarianceEstimate
        """
        index_of_date = get_max_index_before_datetime(self.fit_dates, relevant_date)
        if index_of_date is None:
            return self.cov_list[0]  # forward-fill fallback
        return self.cov_list[index_of_date]


def calculate_covariance_matrices(
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    current_prices_dict: dict
) -> covarianceList:
    """
    Compute a time series of covariance matrices using:
        - weekly percentage returns
        - exponentially weighted correlations
        - shrinkage toward off-diagonal prior
        - conversion to covariance using stdev

    Returns
    -------
    covarianceList
    """
    weekly_df = get_weekly_df_of_percentage_returns(
        adjusted_prices_dict=adjusted_prices_dict,
        current_prices_dict=current_prices_dict
    )

    exp_correlations = calculate_exponential_correlations(weekly_df)
    weekly_index = weekly_df.index

    list_of_cov = [
        calculate_covariance_matrix_at_date(
            relevant_date,
            std_dev_dict=std_dev_dict,
            exp_correlations=exp_correlations
        )
        for relevant_date in weekly_index
    ]

    return covarianceList(cov_list=list_of_cov, fit_dates=weekly_index)


def get_weekly_df_of_percentage_returns(adjusted_prices_dict: dict,
                                        current_prices_dict: dict) -> pd.DataFrame:
    """
    Compute weekly percentage returns for each asset.

    Returns
    -------
    pd.DataFrame
    """
    weekly_idx = get_common_weekly_index(adjusted_prices_dict)
    instruments = list(adjusted_prices_dict.keys())

    returns_dict = {
        instr: calculate_weekly_percentage_returns(
            adjusted_price=adjusted_prices_dict[instr],
            current_price=current_prices_dict[instr],
            weekly_common_index=weekly_idx
        )
        for instr in instruments
    }

    df = pd.concat(returns_dict, axis=1)
    df.columns = instruments
    return df


def calculate_weekly_percentage_returns(adjusted_price: pd.Series,
                                        current_price: pd.Series,
                                        weekly_common_index: list) -> pd.Series:
    """
    Weekly percentage returns = diff(weekly_adj_price) / lagged_current_price

    Returns
    -------
    pd.Series
    """
    adjusted_price_index = adjusted_price.index
    current_price_index = current_price.index
    adjusted_price.index = pd.to_datetime(adjusted_price.index)
    current_price.index = pd.to_datetime(current_price.index)
    weekly_adj = adjusted_price.reindex(weekly_common_index, method="ffill")
    weekly_cur = current_price.reindex(weekly_common_index, method="ffill")
    
    adjusted_price.index = adjusted_price_index
    current_price.index = current_price_index

    price_changes = weekly_adj.diff()
    pct_changes = price_changes / weekly_cur.shift(1)
    return pct_changes


def calculate_exponential_correlations(weekly_df_of_percentage_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute exponentially weighted rolling correlations (pairwise).

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame where level 0 = date, level 1 = asset.
    """
    return weekly_df_of_percentage_returns.ewm(
        span=25,
        min_periods=3,
        ignore_na=True
    ).corr(pairwise=True)


def calculate_covariance_matrix_at_date(relevant_date: datetime,
                                        std_dev_dict: dict,
                                        exp_correlations: pd.DataFrame) -> covarianceEstimate:
    """
    Build covariance matrix at a given date using:
        - exponential correlations
        - shrinkage
        - stdevs
        - correlation→covariance conversion

    Returns
    -------
    covarianceEstimate
    """
    columns = list(std_dev_dict.keys())
    corr_estimate = get_correlation_estimate_at_date(
        relevant_date,
        columns=columns,
        exp_correlations=exp_correlations
    )

    corr_estimate = corr_estimate.shrink_to_offdiag(offdiag=0, shrinkage_corr=0.5)

    stdev_estimate = stdevEstimate(
        get_values_for_date_as_dict(relevant_date, std_dev_dict)
    )

    cov_estimate = calculate_covariance_given_correlation_and_stdev(
        correlation_estimate=corr_estimate,
        stdev_estimate=stdev_estimate
    )

    return cov_estimate


def get_correlation_estimate_at_date(relevant_date: datetime,
                                     columns: list,
                                     exp_correlations: pd.DataFrame) -> correlationEstimate:
    """
    Extract the correlation matrix most recently available before a date.

    Returns
    -------
    correlationEstimate
    """
    size = len(columns)

    corr_values = (
        exp_correlations[exp_correlations.index.get_level_values(0) < relevant_date]
        .tail(size)
        .values
    )

    if corr_values.shape[0] == 0:
        corr_values = np.array([[np.nan] * size] * size)

    return correlationEstimate(values=corr_values, columns=columns)


def get_values_for_date_as_dict(relevant_date: datetime,
                                dict_with_values: dict) -> dict:
    """
    Extract scalar values for the given date for every asset.

    Parameters
    ----------
    dict_with_values : dict
        Mapping {asset: pd.Series}

    Returns
    -------
    dict
        {asset: scalar_or_nan}
    """
    return {
        key_name: get_row_of_series_before_date(ts_series, relevant_date)
        for key_name, ts_series in dict_with_values.items()
    }


def get_row_of_series_before_date(series: pd.Series, relevant_date: datetime):
    """
    Return the value of a time series immediately before a given date.

    Returns
    -------
    float
        Value or NaN if none available.
    """
    series_index = pd.to_datetime(series.index)
    index_point = get_max_index_before_datetime(series_index, relevant_date)
    if index_point is None:
        return np.nan
    return series.values[index_point]


def get_max_index_before_datetime(index, date_point):
    """
    Return index location of the last element strictly before date_point.

    Returns
    -------
    int or None
    """
    count = index[index < date_point].size
    if count == 0:
        return None
    return count - 1


def get_common_index(some_dict: dict) -> list:
    """
    Compute common date index of all series in dict, restricted to DATA_START.

    Returns
    -------
    list
    """
    all_data = pd.concat(some_dict, axis=1)
    all_data.index = pd.to_datetime(all_data.index)
    all_data = all_data[DATA_START:]
    return all_data.index


def get_common_weekly_index(some_dict: dict) -> list:
    """
    Compute common weekly index for all series.

    Returns
    -------
    list
    """

    all_data = pd.concat(some_dict, axis=1)
    all_data.index = pd.to_datetime(all_data.index)
    all_data.resample("7D").last()
    all_data = all_data[DATA_START:]
    
    return all_data.index


def calculate_covariance_given_correlation_and_stdev(
        correlation_estimate: correlationEstimate,
        stdev_estimate: stdevEstimate
    ) -> covarianceEstimate:
    """
    Convert correlation matrix + stdev vector → covariance matrix.

    Handles:
        - asset alignment
        - missing assets inserted with NaN blocks

    Returns
    -------
    covarianceEstimate
    """
    all_assets = set(correlation_estimate.columns) | set(stdev_estimate.list_of_keys())

    assets_with_data = list(
        set(correlation_estimate.assets_with_data())
        & set(stdev_estimate.assets_with_data())
    )

    assets_without_data = list(all_assets - set(assets_with_data))

    aligned_stdev_list = stdev_estimate.list_in_key_order(assets_with_data)

    aligned_corr = correlation_estimate.subset(assets_with_data)

    cov_np = sigma_from_corr_and_std(aligned_stdev_list, aligned_corr.values)

    cov_with_data = covarianceEstimate(cov_np, columns=assets_with_data)

    final_cov = cov_with_data.add_assets_with_nan_values(assets_without_data)

    return final_cov


def sigma_from_corr_and_std(stdev_list: list, corrmatrix: np.array):
    """
    Convert correlation matrix C and stdev vector σ to covariance:

        Σ = diag(σ) · C · diag(σ)

    Returns
    -------
    np.array
    """
    diag = np.diag(stdev_list)
    return diag.dot(corrmatrix).dot(diag)
