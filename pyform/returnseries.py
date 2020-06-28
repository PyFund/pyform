from typing import Optional

import math
import pandas as pd

from pyform.timeseries import TimeSeries


class ReturnSeries(TimeSeries):
    def __init__(self, df):

        super().__init__(df)

    @staticmethod
    def _compound_geometric(returns: pd.Series) -> float:
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

    @staticmethod
    def _compound_arithmetic(returns: pd.Series) -> float:
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

    @staticmethod
    def _compound_continuous(returns: pd.Series) -> float:
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

    def to_freq(self, freq: str, method: str) -> pd.DataFrame:
        """Converts return series to a different (and lower) frequency.

        Args:
            freq: frequency to convert the return series to.
                Available options can be found `here <https://tinyurl.com/t78g6bh>`_.
            method: compounding method when converting to lower frequency.

                * 'geometric': geometric compounding (1+r1) * (1+r2) - 1
                * 'arithmetic': arithmetic compounding r1 + r2
                * 'continuous': continous compounding exp(r1+r2) - 1

        Returns:
            pd.DataFrame: return series in desired frequency
        """

        if method not in ["arithmetic", "geometric", "continuous"]:
            raise ValueError(
                "Method should be one of 'geometric', 'arithmetic' or 'continuous'"
            )

        compound = {
            "arithmetic": self._compound_arithmetic,
            "geometric": self._compound_geometric,
            "continuous": self._compound_continuous,
        }

        return self.df.groupby(pd.Grouper(freq=freq)).agg(compound[method])

    def to_week(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to weekly frequency.

        Args:
            method: compounding method. Defaults to "geometric".

                * 'geometric': geometric compounding (1+r1) * (1+r2) - 1
                * 'arithmetic': arithmetic compounding r1 + r2
                * 'continuous': continous compounding exp(r1+r2) - 1

        Returns:
            pd.DataFrame: return series, in weekly frequency
        """

        return self.to_freq("W", method)

    def to_month(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to monthly frequency.

        Args:
            method: compounding method. Defaults to "geometric".

                * 'geometric': geometric compounding (1+r1) * (1+r2) - 1
                * 'arithmetic': arithmetic compounding r1 + r2
                * 'continuous': continous compounding exp(r1+r2) - 1

        Returns:
            pd.DataFrame: return series, in monthly frequency
        """

        return self.to_freq("M", method)

    def to_quarter(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to quarterly frequency.

        Args:
            method: compounding method. Defaults to "geometric".

                * 'geometric': geometric compounding (1+r1) * (1+r2) - 1
                * 'arithmetic': arithmetic compounding r1 + r2
                * 'continuous': continous compounding exp(r1+r2) - 1

        Returns:
            pd.DataFrame: return series, in quarterly frequency
        """

        return self.to_freq("Q", method)

    def to_year(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to annual frequency.

        Args:
            method: compounding method. Defaults to "geometric".

                * 'geometric': geometric compounding (1+r1) * (1+r2) - 1
                * 'arithmetic': arithmetic compounding r1 + r2
                * 'continuous': continous compounding exp(r1+r2) - 1

        Returns:
            pd.DataFrame: return series, in annual frequency
        """

        return self.to_freq("Y", method)
