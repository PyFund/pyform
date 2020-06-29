import logging

log = logging.getLogger(__name__)

import copy
import math
import pandas as pd
from typing import Callable, Optional, Union
from pyform.timeseries import TimeSeries


class ReturnSeries(TimeSeries):
    """A return series. It should be datetime indexed and
       has one column of returns data.
    """

    def __init__(self, series, name: Optional[str] = None):

        super().__init__(series)

        self.benchmark = dict()
        self.risk_free = dict()

        if name is None:
            self.name = self.series.columns[0]
        else:
            self.name = name

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

    def _compound(self, method: str) -> Callable:

        compound = {
            "arithmetic": self._compound_arithmetic,
            "geometric": self._compound_geometric,
            "continuous": self._compound_continuous,
        }

        return compound[method]

    def to_period(self, freq: str, method: str) -> pd.DataFrame:
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

        if freq == "D":
            # Use businessness days for all return series
            freq = "B"

        try:
            assert self._freq_compare(freq, self.freq)
        except AssertionError:
            raise ValueError(
                "Cannot convert to higher frequency. "
                f"target={freq}, current={self.freq}"
            )

        if method not in ["arithmetic", "geometric", "continuous"]:
            raise ValueError(
                "Method should be one of 'geometric', 'arithmetic' or 'continuous'"
            )

        return self.series.groupby(pd.Grouper(freq=freq)).agg(self._compound(method))

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

        return self.to_period("W", method)

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

        return self.to_period("M", method)

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

        return self.to_period("Q", method)

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

        return self.to_period("Y", method)

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
        self.benchmark[name] = copy.deepcopy(benchmark)

    def add_risk_free(self, risk_free: "ReturnSeries", name: Optional[str] = None):
        """Add a risk free rate for the return series. A benchmark is useful and needed
           in order to calculate:

                * 'sharpe ratio'


        Args:
            risk_free: A risk free rate. Should be a ReturnSeries object.
            name: name of the risk free rate. This will be used to display results.
                Defaults to "None", which will use the column name of the risk free
                rate.
        """

        if name is None:
            name = risk_free.series.columns[0]

        log.info(f"Adding risk free rate. name={name}")
        self.risk_free[name] = copy.deepcopy(risk_free)

    def get_corr(
        self,
        freq: Optional[str] = "M",
        method: Optional[str] = "pearson",
        compound_method: Optional[str] = "geometric",
        meta: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Calculates correlation of the return series with its benchmarks

        Args:
            freq: Returns are converted to the same frequency before correlation
                is compuated. Defaults to "M".
            method: {'pearson', 'kendall', 'spearman'}. Defaults to "pearson".

                * pearson : standard correlation coefficient
                * kendall : Kendall Tau correlation coefficient
                * spearman : Spearman rank correlation

            compound_method: {'geometric', 'arithmetic', 'continuous'}.
                Defaults to "geometric".

            meta: whether to include meta data in output. Defaults to False.
                Available meta are:

                * freq: frequency used to compute correlation
                * method: method used to compute correlation
                * start: start date for calculating correlation
                * end: end date for calculating correlation
                * total: total number of data points in returns series
                * skipped: number of data points skipped when computing correlation

        Raises:
            ValueError: when no benchmark is set

        Returns:
            pd.DataFrame: correlation results with the following columns

                * benchmark: name of the benchmark
                * field: name of the field. In this case, it is 'correlation' for all
                * value: correlation value

            Data described in meta will also be available in the returned DataFrame if
            meta is set to True.
        """

        if not len(self.benchmark) > 0:
            raise ValueError("Correlation needs at least one benchmark.")

        ret = self.to_period(freq=freq, method=compound_method)
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
                bm_ret = benchmark.to_period(freq=freq, method=compound_method)

                # Join returns and benchmark to calculate correlation
                df = ret.join(bm_ret, on="datetime", how="inner")

                # Add correlation to list
                corr.append(df.corr(method).iloc[0, 1])

                # Add benchmark to list of benchmark names
                bm_names.append(name)

                if meta:
                    # Add start and end date used to compute correlation
                    start.append(benchmark.start)
                    end.append(benchmark.end)

                    # Add number of rows skipped in calculation
                    skipped.append(n_ret - len(df.index))

            except Exception as e:  # pragma: no cover

                log.error(f"Cannot compute correlation: benchmark={name}: {e}")
                pass

        if meta:

            result = pd.DataFrame(
                data={
                    "name": bm_names,
                    "field": "correlation",
                    "value": corr,
                    "freq": freq,
                    "method": method,
                    "start": start,
                    "end": end,
                    "total": n_ret,
                    "skipped": skipped,
                }
            )

        else:

            result = pd.DataFrame(
                data={"name": bm_names, "field": "correlation", "value": corr}
            )

        return result

    def get_total_return(
        self,
        include_bm: Optional[bool] = True,
        method: Optional[str] = "geometric",
        meta: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Compute total return of the series

        Args:
            include_bm: whether to compute total return for benchmarks as well.
                Defaults to True.
            method: method to use when compounding total return.
                Defaults to "geometric".
            meta: whether to include meta data in output. Defaults to False.
                Available meta are:

                * method: method used to compound total return
                * start: start date for calculating total return
                * end: end date for calculating total return

        Returns:
            pd.DataFrame: total return results with the following columns

                * name: name of the series
                * field: name of the field. In this case, it is 'total return' for all
                * value: total return value, in decimals

            Data described in meta will also be available in the returned DataFrame if
            meta is set to True.
        """

        # Columns in the returned dataframe
        names = []
        total_return = []
        start = []
        end = []

        names.append(self.name)
        total_return.append(self._compound(method)(self.series.iloc[:, 0]))

        if meta:
            start.append(self.start)
            end.append(self.end)

        if include_bm:
            for name, benchmark in self.benchmark.items():

                try:

                    # Modify benchmark so it's in the same timerange as the
                    # returns series
                    benchmark = self._normalize_daterange(benchmark)
                    names.append(name)
                    total_return.append(
                        self._compound(method)(benchmark.series.iloc[:, 0])
                    )

                    if meta:
                        start.append(benchmark.start)
                        end.append(benchmark.end)

                except Exception as e:  # pragma: no cover

                    log.error(f"Cannot compute total return: benchmark={name}: {e}")
                    pass

        if meta:

            result = pd.DataFrame(
                data={
                    "name": names,
                    "field": "total return",
                    "value": total_return,
                    "method": method,
                    "start": start,
                    "end": end,
                }
            )

        else:

            result = pd.DataFrame(
                data={"name": names, "field": "total return", "value": total_return}
            )

        return result

    def get_annualized_return(
        self,
        method: Optional[str] = "geometric",
        include_bm: Optional[bool] = True,
        meta: Optional[bool] = False,
    ):
        """Compute annualized return of the series

        Args:
            method: method to use when compounding return. Defaults to "geometric".
            include_bm: whether to compute annualized return for benchmarks as well.
                Defaults to True.
            meta: whether to include meta data in output. Defaults to False.
                Available meta are:

                * method: method used to compound annualized return
                * start: start date for calculating annualized return
                * end: end date for calculating annualized return

        Returns:
            pd.DataFrame: annualized return results with the following columns

                * name: name of the series
                * field: name of the field. In this case, it is 'annualized return'
                    for all
                * value: annualized return value, in decimals

            Data described in meta will also be available in the returned DataFrame if
            meta is set to True.
        """

        result = self.get_total_return(method=method, include_bm=include_bm, meta=True)
        result["field"] = "annualized return"

        # find number of days
        one_day = pd.to_timedelta(1, unit="D")
        result["days"] = (result["end"] - result["start"]) / one_day

        years = result["days"] / 365.25
        if method == "geometric":
            result["value"] = (1 + result["value"]) ** (1 / years) - 1
        elif method == "arithmetic":
            result["value"] = result["value"] * (1 / years)
        elif method == "continuous":
            result["value"] = (result["value"] + 1).apply(math.log) * (1 / years)

        if meta:
            result = result[["name", "field", "value", "method", "start", "end"]]
        else:
            result = result[["name", "field", "value"]]

        return result

    def get_annualized_volatility(
        self,
        freq: Optional[str] = "M",
        include_bm: Optional[bool] = True,
        method: Optional[str] = "sample",
        compound_method: Optional[str] = "geometric",
        meta: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Compute annualized volatility of the series

        Args:
            freq: Returns are converted to the same frequency before volatility
                is compuated. Defaults to "M".
            include_bm: whether to compute annualized volatility for benchmarks as well.
                Defaults to True.
            method: {'sample', 'population'}. method used to compute volatility
                (standard deviation). Defaults to "sample".
            compound_method: method to use when compounding return.
                Defaults to "geometric".
            meta: whether to include meta data in output. Defaults to False.
                Available meta are:

                * freq: frequency of the series
                * method: method used to compute annualized volatility
                * start: start date for calculating annualized volatility
                * end: end date for calculating annualized volatility

        Returns:
            pd.DataFrame: annualized volatility results with the following columns

                * name: name of the series
                * field: name of the field. In this case, it is 'annualized volatility'
                    for all
                * value: annualized volatility value, in decimals

            Data described in meta will also be available in the returned DataFrame if
            meta is set to True.
        """

        # delta degrees of freedom, used for calculate standard deviation
        ddof = {"sample": 1, "population": 0}[method]

        # Columns in the returned dataframe
        names = []
        ann_vol = []
        start = []
        end = []

        # Convert series to the desired frequency
        ret = self.to_period(freq=freq, method=compound_method)

        # To annualize, after changing the frequency, see how many
        # periods there are in a year
        one_year = pd.to_timedelta(365.25, unit="D")
        years = (self.end - self.start) / one_year
        sample_per_year = len(ret.index) / years

        vol = ret.iloc[:, 0].std(ddof=ddof)
        vol *= math.sqrt(sample_per_year)

        names.append(ret.columns[0])
        ann_vol.append(vol)

        if meta:
            start.append(self.start)
            end.append(self.end)

        if include_bm:
            for name, benchmark in self.benchmark.items():

                try:

                    # Modify benchmark so it's in the same timerange as the
                    # returns series
                    benchmark = self._normalize_daterange(benchmark)
                    bm = benchmark.to_period(freq=freq, method=compound_method)

                    years = (benchmark.end - benchmark.start) / one_year
                    sample_per_year = len(bm.index) / years

                    vol = bm.iloc[:, 0].std(ddof=ddof)
                    vol *= math.sqrt(sample_per_year)

                    names.append(name)
                    ann_vol.append(vol)

                    if meta:
                        start.append(benchmark.start)
                        end.append(benchmark.end)

                except Exception as e:  # pragma: no cover

                    log.error(
                        "Cannot compute annualized volatility: "
                        f"benchmark={name}: {e}"
                    )
                    pass

        if meta:

            result = pd.DataFrame(
                data={
                    "name": names,
                    "field": "annualized volatility",
                    "value": ann_vol,
                    "freq": freq,
                    "method": method,
                    "start": start,
                    "end": end,
                }
            )

        else:

            result = pd.DataFrame(
                data={"name": names, "field": "annualized volatility", "value": ann_vol}
            )

        return result

    def get_sharpe_ratio(
        self,
        freq: Optional[str] = "M",
        risk_free: Optional[Union[float, int, str]] = 0,
        include_bm: Optional[bool] = True,
        compound_method: Optional[str] = "geometric",
        meta: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Compute Sharpe ratio of the series

        Args:
            freq: Returns are converted to the same frequency before Sharpe ratio
                is compuated. Defaults to "M".
            risk_free: the risk free rate to use. Can be a float or a string. If is
                float, use the value as annualized risk free return. Should be given
                in decimals. i.e. 1% annual cash return will be entered as
                ``annualized_return=0.01``. If is string, look for the corresponding
                DataFrame of risk free rate in ``self.risk_free``. ``self.risk_free``
                can be set via the ``add_risk_free()`` class method. Defaults to 0.
            include_bm: whether to compute Sharpe ratio for benchmarks as well.
                Defaults to True.
            compound_method: method to use when compounding return.
                Defaults to "geometric".
            meta: whether to include meta data in output. Defaults to False.
                Available meta are:

                * freq: frequency of the series
                * risk_free: the risk free rate used
                * start: start date for calculating Sharpe ratio
                * end: end date for calculating Sharpe ratio

        Returns:
            pd.DataFrame: Sharpe ratio with the following columns

                * names: name of the series
                * field: name of the field. In this case, it is 'Sharpe ratio'
                    for all
                * value: Shapre ratio value

            Data described in meta will also be available in the returned DataFrame if
            meta is set to True.
        """

        # create risk free rate
        if isinstance(risk_free, str):
            try:
                rf = self.risk_free[risk_free]
            except KeyError:
                raise ValueError(f"Risk free rate is not set: risk_free={risk_free}")
        elif isinstance(risk_free, float) or isinstance(risk_free, int):
            rf = CashSeries.constant(risk_free, self.start, self.end)
        else:
            raise TypeError(
                "Risk free should be str, float, or int." f"received={type(risk_free)}"
            )

        # create sharpe for main series
        names = []
        sharpe = []
        start = []
        end = []
        risk_free = []

        rf_name = rf.series.columns[0]

        # Make sure the start date is the max of start date of rf and returns,
        # and end date is the min of end date of rf and returns
        start_date = max(rf.start, self.start)
        end_date = min(rf.end, self.end)

        rf_use = copy.deepcopy(rf)
        ret_use = copy.deepcopy(self)
        rf_use.set_daterange(start_date, end_date)
        ret_use.set_daterange(start_date, end_date)
        series_name = ret_use.series.columns[0]

        # compute excess return over rf rate
        df = ret_use.series.merge(
            rf_use.series, on="datetime", how="outer", sort=True
        ).fillna(0)
        df[series_name] -= df[rf_name]
        df = df.drop(rf_name, axis="columns")

        series = ReturnSeries(df)
        ann_ret = series.get_annualized_return(method=compound_method)["value"][0]
        ann_vol = series.get_annualized_volatility(
            freq=freq, compound_method=compound_method
        )["value"][0]
        ratio = ann_ret / ann_vol

        names.append(self.name)
        sharpe.append(ratio)

        if meta:
            rf_ann = rf_use.get_annualized_return(method=compound_method)["value"][0]
            rf_ann = f"{round(rf_ann*100, 2)}%"
            start.append(series.start)
            end.append(series.end)
            risk_free.append(f"{rf_name}: {rf_ann}")

        if include_bm:
            for name, benchmark in self.benchmark.items():

                try:

                    rf_use = copy.deepcopy(rf)
                    benchmark = self._normalize_daterange(benchmark)

                    start_date = max(rf.start, benchmark.start)
                    end_date = min(rf.end, benchmark.end)

                    rf_use.set_daterange(start_date, end_date)
                    benchmark.set_daterange(start_date, end_date)
                    bm_name = benchmark.series.columns[0]

                    df = benchmark.series.merge(
                        rf_use.series, on="datetime", how="outer", sort=True
                    ).fillna(0)
                    df[bm_name] -= df[rf_name]
                    df = df.drop(rf_name, axis="columns")

                    series = ReturnSeries(df)
                    ann_ret = series.get_annualized_return(method=compound_method)[
                        "value"
                    ][0]
                    ann_vol = series.get_annualized_volatility(
                        freq=freq, compound_method=compound_method
                    )["value"][0]
                    ratio = ann_ret / ann_vol

                    names.append(name)
                    sharpe.append(ratio)

                    if meta:
                        rf_ann = rf_use.get_annualized_return(method=compound_method)[
                            "value"
                        ][0]
                        rf_ann = f"{round(rf_ann*100, 2)}%"
                        start.append(series.start)
                        end.append(series.end)
                        risk_free.append(f"{rf_name}: {rf_ann}")

                except Exception as e:  # pragma: no cover

                    log.error("Cannot compute sharpe ratio: " f"benchmark={name}: {e}")
                    pass

        if meta:

            result = pd.DataFrame(
                data={
                    "name": names,
                    "field": "sharpe ratio",
                    "value": sharpe,
                    "freq": freq,
                    "risk_free": risk_free,
                    "start": start,
                    "end": end,
                }
            )

        else:

            result = pd.DataFrame(
                data={"name": names, "field": "sharpe ratio", "value": sharpe}
            )

        return result


class CashSeries(ReturnSeries):
    @classmethod
    def constant(
        cls,
        annualized_return: Optional[Union[float, int]] = 0,
        start: Optional[str] = "1980-01-01",
        end: Optional[str] = "2029-12-31",
    ):
        """Create a constant cash daily returns stream

        Args:
            annualized_return: Annualized return of the cash, in decimals.
                i.e. 1% annual cash return will be entered as
                ``annualized_return=0.01``. Defaults to 0.

        Returns:
            pyform.CashSeries: constant return cash stream
        """

        dates = pd.date_range(start=start, end=end, freq="B")

        one_year = pd.to_timedelta(365.25, unit="D")
        years = (dates[-1] - dates[1]) / one_year
        sample_per_year = len(dates) / years

        # TODO: add other compounding method
        daily_ret = 1 + annualized_return
        daily_ret **= 1 / sample_per_year
        daily_ret -= 1

        daily_ret = pd.DataFrame(
            data={"date": dates, f"cash_{annualized_return}": daily_ret}
        )

        return cls(daily_ret)

    @classmethod
    def read_fred_libor_1m(cls):
        """Create one month libor daily returns from fred data

        Returns:
            pyform.CashSeries: one month libor daily returns
        """

        # Load St.Louis Fed Data
        libor1m = pd.read_csv(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USD1MTD156N"
        )

        # Format Data
        libor1m.columns = ["date", "LIBOR_1M"]
        libor1m = libor1m[libor1m["LIBOR_1M"] != "."]
        libor1m["LIBOR_1M"] = libor1m["LIBOR_1M"].astype(float)
        libor1m["LIBOR_1M"] = libor1m["LIBOR_1M"] / 100

        # Create Return Series
        libor1m = cls(libor1m)

        # Daily Value is in Annualized Form, Change it to Daily Return
        # TODO: See if this should be continous compounding
        one_year = pd.to_timedelta(365.25, unit="D")
        years = (libor1m.end - libor1m.start) / one_year
        sample_per_year = len(libor1m.series.index) / years
        libor1m._series["LIBOR_1M"] += 1
        libor1m._series["LIBOR_1M"] **= 1 / sample_per_year
        libor1m._series["LIBOR_1M"] -= 1
        libor1m.series = libor1m._series.copy()

        return libor1m
