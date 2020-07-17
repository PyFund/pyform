import logging

log = logging.getLogger(__name__)

import copy
import pandas as pd
from typing import Optional, Union, Dict
from pyform.timeseries import TimeSeries
from pyform.returns.compound import compound, ret_to_period, cumseries
from pyform.returns.metrics import calc_ann_vol, calc_ann_ret
from pyform.util.freq import is_lower_freq, calc_samples_per_year


class ReturnSeries(TimeSeries):
    """A return series that's datetime indexed and has one column of returns data
    """

    def __init__(self, series, name: Optional[str] = None):

        super().__init__(series)

        self.benchmark = dict()
        self.risk_free = dict()

        if name is None:
            self.name = self.series.columns[0]
        else:
            self.name = name

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

        # Use businessness days for all return series
        if freq == "D":
            freq = "B"

        if freq == self.freq:
            return self.series

        # make sure it's not converting to a higher frequency
        # e.g. trying to convert a monthly series into daily
        try:
            assert is_lower_freq(freq, self.freq)
        except AssertionError:
            raise ValueError(
                "Cannot convert to higher frequency. "
                f"target={freq}, current={self.freq}"
            )

        return ret_to_period(self.series, freq, method)

    def to_week(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to weekly frequency.

        Args:
            method: compounding method. Defaults to "geometric".

        Returns:
            pd.DataFrame: return series, in weekly frequency
        """

        return self.to_period("W", method)

    def to_month(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to monthly frequency.

        Args:
            method: compounding method. Defaults to "geometric".

        Returns:
            pd.DataFrame: return series, in monthly frequency
        """

        return self.to_period("M", method)

    def to_quarter(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to quarterly frequency.

        Args:
            method: compounding method. Defaults to "geometric".

        Returns:
            pd.DataFrame: return series, in quarterly frequency
        """

        return self.to_period("Q", method)

    def to_year(self, method: Optional[str] = "geometric") -> pd.DataFrame:
        """Converts return series to annual frequency.

        Args:
            method: compounding method. Defaults to "geometric".

        Returns:
            pd.DataFrame: return series, in annual frequency
        """

        return self.to_period("Y", method)

    def add_bm(self, benchmark: "ReturnSeries", name: Optional[str] = None):
        """Adds a benchmark for the return series.

        A benchmark is useful and needed in order to calculate:

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

    def add_rf(self, risk_free: "ReturnSeries", name: Optional[str] = None):
        """Adds a risk free rate for the return series.

        A risk free rate is useful and needed in order to calculate:

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

        # Columns in the returned dataframe
        names, corr, start, end, used = ([] for i in range(5))

        # Convert return
        ret = self.to_period(freq=freq, method=compound_method)

        for name, benchmark in self.benchmark.items():

            try:

                # keep record of start and so they can be reset later
                bm_start = benchmark.start
                bm_end = benchmark.end

                # modify benchmark so it's in the same timerange as the returns series
                self.align_daterange(benchmark)

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
                names.append(name)

                if meta:
                    start.append(benchmark.start)  # start date of data used
                    end.append(benchmark.end)  # end date of data used
                    used.append(len(df.index))  # number of rows used in calculation

                # Reset bm daterange
                benchmark.set_daterange(bm_start, bm_end)

            except Exception as e:  # pragma: no cover

                log.error(f"Cannot compute correlation: benchmark={name}: {e}")
                pass

        if meta:

            result = pd.DataFrame(
                data={
                    "name": names,
                    "field": "correlation",
                    "value": corr,
                    "freq": freq,
                    "method": method,
                    "start": start,
                    "end": end,
                    "total": len(ret.index),
                    "used": used,
                }
            )

        else:

            result = pd.DataFrame(
                data={"name": names, "field": "correlation", "value": corr}
            )

        return result

    def get_tot_ret(
        self,
        include_bm: Optional[bool] = True,
        method: Optional[str] = "geometric",
        meta: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Computes total return of the series

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
        names, total_return, start, end = ([] for i in range(4))

        run_name, run_data = [self.name], [self]

        if include_bm:
            run_name += list(self.benchmark.keys())
            run_data += list(self.benchmark.values())

        for name, series in zip(run_name, run_data):

            try:

                # keep record of start and so they can be reset later
                series_start, series_end = series.start, series.end

                # modify series so it's in the same timerange as the main series
                self.align_daterange(series)
                tot_ret = compound(method)(series.series.iloc[:, 0])

                names.append(name)
                total_return.append(tot_ret)

                if meta:
                    start.append(series.start)
                    end.append(series.end)

                series.set_daterange(series_start, series_end)

            except Exception as e:  # pragma: no cover

                log.error(f"Cannot compute total return: name={name}: {e}")
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

    def get_index_series(
        self,
        freq: Optional[str] = "M",
        method: Optional[str] = "geometric",
        include_bm: Optional[bool] = True,
    ) -> Dict[str, pd.DataFrame]:

        # Store result in dictionary
        result = dict()

        run_name, run_data = [self.name], [self]

        if include_bm:
            run_name += list(self.benchmark.keys())
            run_data += list(self.benchmark.values())

        for name, series in zip(run_name, run_data):

            # keep record of start and so they can be reset later
            series_start, series_end = series.start, series.end

            # modify series so it's in the same timerange as the main series
            self.align_daterange(series)

            # compute rolling annualized volatility
            ret = series.to_period(freq=freq, method=method)

            # store result in dictionary
            result[name] = ret.apply(cumseries(method))

            # reset series date range
            series.set_daterange(series_start, series_end)

        return result

    def get_ann_ret(
        self,
        method: Optional[str] = "geometric",
        include_bm: Optional[bool] = True,
        meta: Optional[bool] = False,
    ):
        """Computes annualized return of the series

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

        # Store result in dictionary
        names, ann_return, start, end = ([] for i in range(4))

        run_name, run_data = [self.name], [self]

        if include_bm:
            run_name += list(self.benchmark.keys())
            run_data += list(self.benchmark.values())

        for name, series in zip(run_name, run_data):

            # keep record of start and so they can be reset later
            series_start, series_end = series.start, series.end

            # modify series so it's in the same timerange as the main series
            self.align_daterange(series)

            # compute rolling annualized return
            ann_ret = calc_ann_ret(series.series, method)

            names.append(name)
            ann_return.append(ann_ret)

            if meta:
                start.append(series.start)
                end.append(series.end)

            # reset series date range
            series.set_daterange(series_start, series_end)

        if meta:

            result = pd.DataFrame(
                data={
                    "name": names,
                    "field": "annualized return",
                    "value": ann_return,
                    "method": method,
                    "start": start,
                    "end": end,
                }
            )

        else:

            result = pd.DataFrame(
                data={"name": names, "field": "annualized return", "value": ann_return}
            )

        return result

    def get_ann_vol(
        self,
        freq: Optional[str] = "M",
        include_bm: Optional[bool] = True,
        method: Optional[str] = "sample",
        compound_method: Optional[str] = "geometric",
        meta: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Computes annualized volatility of the series

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

        # Columns in the returned dataframe
        names, ann_vol, start, end = ([] for i in range(4))

        run_name, run_data = [self.name], [self]

        if include_bm:
            run_name += list(self.benchmark.keys())
            run_data += list(self.benchmark.values())

        for name, series in zip(run_name, run_data):

            try:

                # keep record of start and so they can be reset later
                series_start, series_end = series.start, series.end

                # modify series so it's in the same timerange as the main series
                self.align_daterange(series)

                # Convert return to desired frequency
                ret = series.to_period(freq=freq, method=compound_method)
                samples_per_year = calc_samples_per_year(
                    len(ret.index), series.start, series.end
                )
                vol = calc_ann_vol(ret, method, samples_per_year)

                names.append(name)
                ann_vol.append(vol)

                if meta:
                    start.append(series.start)
                    end.append(series.end)

                series.set_daterange(series_start, series_end)

            except Exception as e:  # pragma: no cover

                log.error(f"Cannot compute annualized volatility: name={name}: {e}")
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

    def get_sharpe(
        self,
        freq: Optional[str] = "M",
        risk_free: Optional[Union[float, int, str]] = 0,
        include_bm: Optional[bool] = True,
        compound_method: Optional[str] = "geometric",
        meta: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Computes Sharpe ratio of the series

        Args:
            freq: Returns are converted to the same frequency before Sharpe ratio
                is compuated. Defaults to "M".
            risk_free: the risk free rate to use. Can be a float or a string. If is
                float, use the value as annualized risk free return. Should be given
                in decimals. i.e. 1% annual cash return will be entered as
                ``annualized_return=0.01``. If is string, look for the corresponding
                DataFrame of risk free rate in ``self.risk_free``. ``self.risk_free``
                can be set via the ``add_rf()`` class method. Defaults to 0.
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
            try:
                rf = self.risk_free[f"cash_{risk_free}"]
            except KeyError:
                rf = CashSeries.constant(risk_free, self.start, self.end)
                self.add_rf(rf, f"cash_{risk_free}")
        else:
            raise TypeError(
                "Risk free should be str, float, or 0." f"received={type(risk_free)}"
            )

        # create sharpe for main series
        names, sharpe, start, end, risk_free = ([] for i in range(5))

        # get column name of risk free rate
        rf_name = rf.series.columns[0]

        run_name, run_data = [self.name], [self]

        if include_bm:
            run_name += list(self.benchmark.keys())
            run_data += list(self.benchmark.values())

        for name, series in zip(run_name, run_data):

            try:

                # keep record of start and so they can be reset later
                series_start, series_end = series.start, series.end

                # modify series so it's in the same timerange as the main series
                self.align_daterange(series)

                # get name of the series
                name = series.series.columns[0]

                # use the narrowest date range between series and risk free rate
                start_date = max(rf.start, series.start)
                end_date = min(rf.end, series.end)
                rf.set_daterange(start_date, end_date)
                series.set_daterange(start_date, end_date)

                df = series.series.merge(
                    rf.series, on="datetime", how="outer", sort=True
                ).fillna(0)
                df[name] -= df[rf_name]
                df = df.drop(rf_name, axis="columns")

                exccess_series = ReturnSeries(df)
                ann_excess_ret = exccess_series.get_ann_ret(
                    method=compound_method, include_bm=False
                )["value"][0]
                ann_series_vol = series.get_ann_vol(
                    freq=freq, compound_method=compound_method, include_bm=False
                )["value"][0]
                ratio = ann_excess_ret / ann_series_vol

                names.append(name)
                sharpe.append(ratio)

                if meta:
                    rf_ann = rf.get_ann_ret(method=compound_method, include_bm=False)[
                        "value"
                    ][0]
                    rf_ann = f"{round(rf_ann*100, 2)}%"
                    risk_free.append(f"{rf_name}: {rf_ann}")
                    start.append(series.start)
                    end.append(series.end)

                series.set_daterange(series_start, series_end)
                rf.reset()

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

    def get_rolling_tot_ret(
        self,
        window: Optional[int] = 36,
        freq: Optional[str] = "M",
        include_bm: Optional[bool] = True,
        method: Optional[str] = "geometric",
    ) -> Dict[str, pd.DataFrame]:
        """Computes rolling total return of the series

        Args:
            window: the rolling window. Defaults to 36.
            freq: frequency of the series. Defaults to "M".
            include_bm: whether to compute rolling total return for
                benchmarks as well. Defaults to True.
            method: method to use when compounding total return.
                Defaults to "geometric".


        Returns:
            Dict[pd.DataFrame]: dictionary of rolling total returns

                * key: name of the series
                * value: rolling total returns, in a datetime indexed pandas dataframe
        """

        # Store result in dictionary
        result = dict()

        run_name, run_data = [self.name], [self]

        if include_bm:
            run_name += list(self.benchmark.keys())
            run_data += list(self.benchmark.values())

        for name, series in zip(run_name, run_data):

            # keep record of start and so they can be reset later
            series_start, series_end = series.start, series.end

            # modify series so it's in the same timerange as the main series
            self.align_daterange(series)

            # compute rolling total return
            ret = series.to_period(freq=freq, method=method)
            roll_result = ret.rolling(window).apply(compound(method))
            roll_result = roll_result.dropna()

            # store result in dictionary
            result[name] = roll_result

            # reset series date range
            series.set_daterange(series_start, series_end)

        return result

    def get_rolling_ann_ret(
        self,
        window: Optional[int] = 36,
        freq: Optional[str] = "M",
        method: Optional[str] = "geometric",
        include_bm: Optional[bool] = True,
    ) -> Dict[str, pd.DataFrame]:
        """Computes rolling annualized returns of the series

        Args:
            window: the rolling window. Defaults to 36.
            freq: Returns are converted to the same frequency before annualized
                return is compuated. Defaults to "M".
            compound_method: method to use when compounding return to desired
                frequency. Defaults to "geometric".
            include_bm: whether to compute rolling annualized returns for
                benchmarks as well. Defaults to True.

        Returns:
            Dict[pd.DataFrame]: dictionary of rolling annualized returns

                * key: name of the series
                * value: rolling annualized returns, in a datetime indexed pandas
                    dataframe
        """

        # Store result in dictionary
        result = dict()

        run_name, run_data = [self.name], [self]

        if include_bm:
            run_name += list(self.benchmark.keys())
            run_data += list(self.benchmark.values())

        for name, series in zip(run_name, run_data):

            # keep record of start and so they can be reset later
            series_start, series_end = series.start, series.end

            # modify series so it's in the same timerange as the main series
            self.align_daterange(series)

            # compute rolling annualized return
            ret = series.to_period(freq=freq, method=method)

            # TODO: annualization can be more precise for the start and end period
            # number of days in each period
            freq_year = {"M": 12, "Q": 4, "Y": 1}

            if freq in ["D", "B"]:
                years = None
            else:
                years = window / freq_year[freq]

            roll_result = ret.rolling(window).apply(
                lambda x: calc_ann_ret(x, method, years)
            )
            roll_result = roll_result.dropna()

            # store result in dictionary
            result[name] = roll_result

            # reset series date range
            series.set_daterange(series_start, series_end)

        return result

    def get_rolling_ann_vol(
        self,
        window: Optional[int] = 36,
        freq: Optional[str] = "M",
        method: Optional[str] = "sample",
        include_bm: Optional[bool] = True,
        compound_method: Optional[str] = "geometric",
    ) -> Dict[str, pd.DataFrame]:
        """Computes rolling volatility (standard deviation) of the series

        Args:
            window: the rolling window. Defaults to 36.
            freq: Returns are converted to the same frequency before volatility
                is compuated. Defaults to "M".
            method: {'sample', 'population'}. method used to compute volatility
                (standard deviation). Defaults to "sample".
            include_bm: whether to compute rolling volatility for
                benchmarks as well. Defaults to True.
            compound_method: method to use when compounding return.
                Defaults to "geometric".

        Returns:
            Dict[pd.DataFrame]: dictionary of rolling annualized volatilities

                * key: name of the series
                * value: rolling annualized volatilities, in a datetime indexed
                    pandas dataframe
        """

        # Store result in dictionary
        result = dict()

        run_name, run_data = [self.name], [self]

        if include_bm:
            run_name += list(self.benchmark.keys())
            run_data += list(self.benchmark.values())

        for name, series in zip(run_name, run_data):

            # keep record of start and so they can be reset later
            series_start, series_end = series.start, series.end

            # modify series so it's in the same timerange as the main series
            self.align_daterange(series)

            # compute rolling annualized volatility
            ret = series.to_period(freq=freq, method=compound_method)
            samples_per_year = calc_samples_per_year(
                len(ret.index), series.start, series.end
            )
            roll_result = ret.rolling(window).apply(
                lambda x: calc_ann_vol(
                    x, method=method, samples_per_year=samples_per_year
                )
            )
            roll_result = roll_result.dropna()

            # store result in dictionary
            result[name] = roll_result

            # reset series date range
            series.set_daterange(series_start, series_end)

        return result


class CashSeries(ReturnSeries):
    @classmethod
    def constant(
        cls,
        annualized_return: Optional[Union[float, int]] = 0,
        start: Optional[str] = "1980-01-01",
        end: Optional[str] = "2029-12-31",
    ):
        """Creates a constant cash daily returns stream

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
        """Creates one month libor daily returns from fred data

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
