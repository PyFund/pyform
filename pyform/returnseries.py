import logging

log = logging.getLogger(__name__)

import copy
import math
import pandas as pd
from typing import Optional
from pyform.timeseries import TimeSeries


class ReturnSeries(TimeSeries):
    """A return series. It should be datetime indexed and
       has one column of returns data.

    Args:
        TimeSeries ([type]): [description]
    """

    def __init__(self, df):

        super().__init__(df)

        self.benchmark = dict()

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
        calculate ``r1 + r2`` + r3

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
        calculate exp(``r1 + r2`` + r3) - 1

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

                * 'geometric': geometric compounding ``(1+r1) * (1+r2) - 1``
                * 'arithmetic': arithmetic compounding ``r1 + r2``
                * 'continuous': continous compounding ``exp(r1+r2) - 1``

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

        return self.series.groupby(pd.Grouper(freq=freq)).agg(compound[method])

    def to_week(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to weekly frequency.

        Args:
            method: compounding method. Defaults to "geometric".

                * 'geometric': geometric compounding ``(1+r1) * (1+r2) - 1``
                * 'arithmetic': arithmetic compounding ``r1 + r2``
                * 'continuous': continous compounding ``exp(r1+r2) - 1``

        Returns:
            pd.DataFrame: return series, in weekly frequency
        """

        return self.to_freq("W", method)

    def to_month(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to monthly frequency.

        Args:
            method: compounding method. Defaults to "geometric".

                * 'geometric': geometric compounding ``(1+r1) * (1+r2) - 1``
                * 'arithmetic': arithmetic compounding ``r1 + r2``
                * 'continuous': continous compounding ``exp(r1+r2) - 1``

        Returns:
            pd.DataFrame: return series, in monthly frequency
        """

        return self.to_freq("M", method)

    def to_quarter(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to quarterly frequency.

        Args:
            method: compounding method. Defaults to "geometric".

                * 'geometric': geometric compounding ``(1+r1) * (1+r2) - 1``
                * 'arithmetic': arithmetic compounding ``r1 + r2``
                * 'continuous': continous compounding ``exp(r1+r2) - 1``

        Returns:
            pd.DataFrame: return series, in quarterly frequency
        """

        return self.to_freq("Q", method)

    def to_year(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to annual frequency.

        Args:
            method: compounding method. Defaults to "geometric".

                * 'geometric': geometric compounding ``(1+r1) * (1+r2) - 1``
                * 'arithmetic': arithmetic compounding ``r1 + r2``
                * 'continuous': continous compounding ``exp(r1+r2) - 1``

        Returns:
            pd.DataFrame: return series, in annual frequency
        """

        return self.to_freq("Y", method)

    def _normalize_daterange(self, series: "ReturnSeries"):

        series = copy.deepcopy(series)
        series.set_daterange(start=self.start, end=self.end)

        return series

    def add_benchmark(self, benchmark: "ReturnSeries", name: Optional[str] = None):
        """Add a benchmark for the return series. A benchmark is useful and needed
           in order to calculate:

                * 'correlation': is the correlation between the return series and
                    the benchmark
                * 'beta': is the CAPM beta between the return series and the benchmark

        Args:
            benchmark: A benchmark. Should be a ReturnSeries object.
            name: name of the benchmark. This will be used to display results. Defaults
                to "None", which will use the column name of the benchmark.
        """

        if name is None:
            name = benchmark.series.columns[0]

        log.info(f"Adding benchmark. name={name}")
        self.benchmark[name] = benchmark

    def get_corr(
        self,
        freq: Optional[str] = "M",
        compound_method: Optional[str] = "geometric",
        meta: Optional[bool] = False,
    ):

        ret = self.to_freq(freq=freq, method=compound_method)
        n_ret = len(ret.index)

        # Columns in the returned dataframe
        bm_names = []
        corr = []
        start = []
        end = []
        skipped = []

        for name, benchmark in self.benchmark.items():

            try:

                # Modify benchmark so it's in the same timerange as the returns series
                benchmark = self._normalize_daterange(benchmark)

                # Convert benchmark to desired frequency
                # note this is done after it's time range has been normalized
                # this is important as otherwise when frequency is changed, we may
                # include additional days in the calculation
                bm_ret = benchmark.to_freq(freq=freq, method=compound_method)

                # Join returns and benchmark to calculate correlation
                df = ret.join(bm_ret, on="datetime", how="inner")

                # Add correlation to list
                corr.append(df.corr().iloc[0, 1])

                # Add benchmark to list of benchmark names
                bm_names.append(name)

                if meta:
                    # Add start and end date used to compute correlation
                    start.append(min(benchmark.start, self.start))
                    end.append(min(benchmark.end, self.end))

                    # Add number of rows skipped in calculation
                    skipped.append(n_ret - len(df.index))

            except Exception as e:

                log.error(f"Cannot compute correlation: benchmark={name}: {e}")
                pass

        if meta:

            result = pd.DataFrame(
                data={
                    "benchmark": bm_names,
                    "field": "correlation",
                    "value": corr,
                    "start": start,
                    "end": end,
                    "total": n_ret,
                    "skipped": skipped,
                }
            )

        else:

            result = pd.DataFrame(
                data={"benchmark": bm_names, "field": "correlation", "value": corr}
            )

        return result
