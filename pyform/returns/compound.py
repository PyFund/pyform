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
    calculate ``r1 + r2`` + r3

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
    calculate exp(``r1 + r2`` + r3) - 1

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
