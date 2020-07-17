import math
import pandas as pd
from typing import Callable


def compound_geometric(returns: pd.Series) -> float:
    """Performs geometric compounding.

    e.g. if there are 3 returns r1, r2, r3,
    calculate (1+r1) * (1+r2) * (1+r3) - 1

    Args:
        returns: pandas series of returns, in decimals.
            i.e. 3% should be expressed as 0.03, not 3.

    Returns:
        float: total compounded return
    """

    return (1 + returns).prod() - 1


def compound_arithmetic(returns: pd.Series) -> float:
    """Performs arithmatic compounding.

    e.g. if there are 3 returns r1, r2, r3,
    calculate r1 + r2 + r3

    Args:
        returns: pandas series of returns, in decimals.
            i.e. 3% should be expressed as 0.03, not 3.

    Returns:
        float: total compounded return
    """

    return sum(returns)


def compound_continuous(returns: pd.Series) -> float:
    """Performs continuous compounding.

    e.g. if there are 3 returns r1, r2, r3,
    calculate exp(r1 + r2 + r3) - 1

    Args:
        returns: pandas series of returns, in decimals.
            i.e. 3% should be expressed as 0.03, not 3.

    Returns:
        float: total compounded return
    """

    return math.exp(sum(returns)) - 1


def compound(method: str) -> Callable:
    """Factory for producing compound functions.

    Args:
        method: method of compounding in the generated function

            * 'geometric': geometric compounding ``(1+r1) * (1+r2) - 1``
            * 'arithmetic': arithmetic compounding ``r1 + r2``
            * 'continuous': continous compounding ``exp(r1+r2) - 1``

    Raises:
        ValueError: when method is not supported.

    Returns:
        Callable: a function that takes a pandas series as its argument, and
            compound it according to the method specified.
    """

    if method not in ["arithmetic", "geometric", "continuous"]:
        raise ValueError(
            "Method should be one of 'geometric', 'arithmetic' or 'continuous'"
        )

    compound = {
        "arithmetic": compound_arithmetic,
        "geometric": compound_geometric,
        "continuous": compound_continuous,
    }

    return compound[method]


def cumseries_geometric(returns: pd.Series) -> pd.Series:
    """Performs geometric compounding to create cumulative index series.

    e.g. if there are 3 returns r1, r2, r3,
    calculate
        (1+r1) - 1,
        (1+r1) * (1+r2) - 1,
        (1+r1) * (1+r2) * (1+r3) - 1

    Args:
        returns: pandas series of returns, in decimals.
            i.e. 3% should be expressed as 0.03, not 3.

    Returns:
        returns: pandas series of cumulative index, in decimals.
    """

    return (1 + returns).cumprod() - 1


def cumseries_arithmetic(returns: pd.Series) -> pd.Series:
    """Performs arithmatic compounding to create cumulative index series.

    e.g. if there are 3 returns r1, r2, r3,
    calculate
        r1
        r1 + r2
        r1 + r2 + r3

    Args:
        returns: pandas series of returns, in decimals.
            i.e. 3% should be expressed as 0.03, not 3.

    Returns:
        returns: pandas series of cumulative index, in decimals.
    """

    return returns.cumsum()


def cumseries_continuous(returns: pd.Series) -> float:
    """Performs continuous compounding to create cumulative index series.

    e.g. if there are 3 returns r1, r2, r3,
    calculate
        exp(r1) - 1
        exp(r1 + r2) - 1
        exp(r1 + r2 + r3) - 1

    Args:
        returns: pandas series of returns, in decimals.
            i.e. 3% should be expressed as 0.03, not 3.

    Returns:
        returns: pandas series of cumulative index, in decimals.
    """

    return returns.cumsum().apply(lambda x: math.exp(x) - 1)


def cumseries(method: str) -> Callable:
    """Factory for producing compound functions.

    Args:
        method: method of compounding in the generated function

            * 'geometric': geometric compounding
            * 'arithmetic': arithmetic compounding
            * 'continuous': continous compounding

    Raises:
        ValueError: when method is not supported.

    Returns:
        Callable: a function that takes a pandas series as its argument, and
            compound it to create a cumulative index sereis according to the
            method specified.
    """

    if method not in ["arithmetic", "geometric", "continuous"]:
        raise ValueError(
            "Method should be one of 'geometric', 'arithmetic' or 'continuous'"
        )

    cumseries = {
        "arithmetic": cumseries_arithmetic,
        "geometric": cumseries_geometric,
        "continuous": cumseries_continuous,
    }

    return cumseries[method]


def ret_to_period(df: pd.DataFrame, freq: str, method: str):
    """Converts return series to a different (and lower) frequency.

    Args:
        df: a time indexed pandas dataframe
        freq: frequency to convert the return series to.
            Available options can be found `here <https://tinyurl.com/t78g6bh>`_.
        method: compounding method when converting to lower frequency.

            * 'geometric': geometric compounding ``(1+r1) * (1+r2) - 1``
            * 'arithmetic': arithmetic compounding ``r1 + r2``
            * 'continuous': continous compounding ``exp(r1+r2) - 1``

    Returns:
        pd.DataFrame: return series in desired frequency
    """
    return df.groupby(pd.Grouper(freq=freq)).agg(compound(method))
