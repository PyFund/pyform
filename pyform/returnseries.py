from typing import Optional

import math
import pandas as pd

from pyform.timeseries import TimeSeries


class ReturnSeries(TimeSeries):
    def __init__(self, df):

        super().__init__(df)

    @staticmethod
    def _compound_geometric(returns: pd.Series):

        return math.prod(1 + returns) - 1

    @staticmethod
    def _compound_arithmetic(returns: pd.Series):

        return sum(returns)

    @staticmethod
    def _compound_continuous(returns: pd.Series):

        return math.exp(sum(returns)) - 1

    def to_freq(self, freq: str, method: str):

        compound = {
            "arithmetic": self._compound_arithmetic,
            "geometric": self._compound_geometric,
            "continuous": self._compound_continuous,
        }

        return self.df.groupby(pd.Grouper(freq=freq)).agg(compound[method])

    def to_week(self, method: Optional[str] = "geometric"):

        return self.to_freq("W", method)

    def to_month(self, method: Optional[str] = "geometric"):

        return self.to_freq("M", method)

    def to_quarter(self, method: Optional[str] = "geometric"):

        return self.to_freq("Q", method)

    def to_year(self, method: Optional[str] = "geometric"):

        return self.to_freq("Y", method)
