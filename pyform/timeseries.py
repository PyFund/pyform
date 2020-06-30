import logging

log = logging.getLogger(__name__)

import pandas as pd
from typing import Optional, Union
from pyform.util.dataframe import set_col_as_datetime_index
from pyform.util.freq import infer_freq


class TimeSeries:
    """TimeSeries is a representation of a form of data that changes with time.

    For a timeseries object, time index should be unique, meaning it would only
    accept a "wide" dataframe and not a "long" dataframe.

       Args:
           df: a dataframe with datetime index, or a 'date'/'datetime' column
    """

    def __init__(self, df: pd.DataFrame):

        df = self._validate_input(df)

        # series is the one we are currently analyzing, while
        # _series stores the initial input. This allows us to
        # go to different timerange and comeback, without losing
        # any information
        self._series = df.copy()
        self._start = min(self._series.index)
        self._end = max(self._series.index)

        # start and end date of the series
        self.series = df.copy()
        self.start = min(self.series.index)
        self.end = max(self.series.index)

        # frequency of the series
        self.freq = infer_freq(self.series)

    @classmethod
    def read_csv(cls, path: str):
        """Creates a time series object from a csv file

        Args:
            path: path to the csv file

        Returns:
            pyform.TimeSeries: a TimeSeries object
        """

        df = pd.read_csv(path)
        return cls(df)

    @classmethod
    def read_excel(cls, path: str, sheet_name: Union[str, int] = 0):
        """Creates a time series object from a Excel file

        Note:
            using this method requires additional dependency openpyxl

        Args:
            path: path to the Excel file
            sheet_name: the index or name of the sheet to read in. Defaults to 0.

        Returns:
            pyform.TimeSeries: a TimeSeries object
        """

        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        return cls(df)

    @classmethod
    def read_db(cls, query: str):

        return NotImplemented

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates the DataFrame's format.

        The dataframe should be a time indexed pandas dataframe,
        or it has one column named as "date" or "datetime".

        Args:
            df: a time indexed pandas dataframe, or a pandas dataframe
            with one column named as "date" or "datetime"

        Raises:
            TypeError: when df is not a pandas dataframe
            ValueError: when there is no datetime index and no
                date or datetime column

        Returns:
            pd.DataFrame: formatted dataframe, with datetime index and
            one column of data
        """

        # Check we are getting a pandas dataframe
        try:
            assert isinstance(df, pd.DataFrame)
        except AssertionError:
            raise TypeError("TimeSeries df argument must be a pandas DataFrame")

        # We are getting a pandas dataframe, and it is datetime indexed.
        # Return it.
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = "datetime"
            return df

        # We are getting a pandas dataframe, but without datetime index.
        # See if one of the columns can be converted to the required datetime index.
        has_datetime = "datetime" in df
        has_date = "date" in df

        try:
            assert has_datetime or has_date
        except AssertionError:
            raise ValueError(
                "TimeSeries df argument without DatetimeIndex "
                "should have a 'date' or 'datetime' column"
            )

        # datetime column is preferred, as the name suggests it also has time in it,
        # which helps make the time series more precise
        if has_datetime:
            return set_col_as_datetime_index(df, "datetime")

        # lastly, use date as index
        if has_date:
            return set_col_as_datetime_index(df, "date")

    def set_daterange(self, start: Optional[str] = None, end: Optional[str] = None):
        """Sets the period of the series we are interested in.

        Args:
            start: the start date, in YYYY-MM-DD HH:MM:SS, hour is optional.
                Defaults to None.
            end: the end date, in YYYY-MM-DD HH:MM:SS, hour is optional.
                Defaults to None.
        """

        if start is not None and end is not None:
            self.series = self._series.copy().loc[start:end]
        elif start is not None:
            self.series = self._series.copy().loc[start:]
        elif end is not None:
            self.series = self._series.copy().loc[:end]

        self.start = min(self.series.index)
        self.end = max(self.series.index)

    def align_daterange(self, series: "TimeSeries"):
        """Aligns daterange of the incoming series with the main series

        Args:
            series: series' to align daterange for
        """

        series.set_daterange(start=self.start, end=self.end)

    def reset(self):
        """Resets TimeSeries to its initial state
        """

        self.series = self._series.copy()
        self.start = min(self.series.index)
        self.end = max(self.series.index)
