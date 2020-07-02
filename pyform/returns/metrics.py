import math
import pandas as pd
from typing import Optional, Union
from pyform.returns.compound import compound
from pyform.util.freq import calc_samples_per_year, calc_timedelta_in_years


def calc_ann_vol(
    series: Union[pd.DataFrame, pd.Series],
    method: str,
    samples_per_year: Optional[float] = None,
) -> float:
    """Computes annualized volatility of a time indexed pandas series

    Args:
        series: a time indexed pandas DataFrame or Series of returns
        method: {'sample', 'population'}. method used to compute volatility
            (standard deviation).
        samples_per_year: Useful when you want to specify how many samples are there
            per year so annualization can be done properly. If None, this will be
            computed by using the series supplied. Defaults to None.

    Returns:
        float: annualized volatility
    """

    # delta degrees of freedom, used for calculate standard deviation
    ddof = {"sample": 1, "population": 0}[method]

    if samples_per_year is None:
        samples_per_year = calc_samples_per_year(
            len(series.index), min(series.index), max(series.index)
        )

    # Compute per period standard deviation
    if isinstance(series, pd.DataFrame):
        returns = series.iloc[:, 0]
    elif isinstance(series, pd.Series):
        returns = series

    vol = returns.std(ddof=ddof)

    # Annualize to annual volatility
    vol *= math.sqrt(samples_per_year)

    return vol


def calc_ann_ret(
    series: Union[pd.DataFrame, pd.Series], method: str, years: Optional[float] = None,
) -> float:
    """Computes annualized return of a time indexed pandas series

    Args:
        series: a time indexed pandas DataFrame or Series of returns
        method: {'geometric', 'arithmetic', 'continuous'}. method used to compound
            returns
        years: Useful when you want to specify how many years are actually
            in the data.

    Returns:
        float: annualized volatility
    """

    if years is None:
        years = calc_timedelta_in_years(min(series.index), max(series.index))

    # Compute per period standard deviation
    if isinstance(series, pd.DataFrame):
        returns = series.iloc[:, 0]
    elif isinstance(series, pd.Series):
        returns = series

    tot_ret = compound(method)(returns)

    if method == "geometric":
        ann_ret = (tot_ret + 1) ** (1 / years) - 1
    elif method == "arithmetic":
        ann_ret = tot_ret * (1 / years)
    elif method == "continuous":
        ann_ret = math.log(tot_ret + 1) * (1 / years)

    return ann_ret
