import logging

log = logging.getLogger(__name__)

import pandas as pd
from typing import Optional


class TimeSeries:
    """TimeSeries is a representation of a form of data that changes with
       time. For a timeseries object, time index should be unique, meaning
       it would only accept a "wide" dataframe and not a "long" dataframe.

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
        self.series = df.copy()

        self.start = min(self.series.index)
        self.end = max(self.series.index)

    @classmethod
    def read_csv(cls, path: str):
        """Create a time series object from a csv file

        Args:
            path: path to the csv file

        Returns:
            pyform.TimeSeries: a TimeSeries object
        """

        df = pd.read_csv(path)
        return cls(df)

    @classmethod
    def read_excel(cls, path: str):

        return NotImplemented

    @classmethod
    def read_db(cls, query: str):

        return NotImplemented

    @staticmethod
    def _set_col_as_datetime_index(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Sets a column in the DataFrame as its datetime index, and name
        the index "datetime"

        Args:
            df: dataframe to set datetime index
            col: column to set as the datetime index for the DataFrame

        Raises:
            ValueError: raised when column cannot be converted to datetime index

        Returns:
            pd.DataFrame: a pandas dataframe with datetime index
        """

        log.info(f"Using {col} column as index.")
        try:
            df = df.set_index(col)
            df.index = pd.to_datetime(df.index)
            df.index.name = "datetime"
            return df
        except Exception as err:
            raise ValueError(f"Error converting '{col}' to index: {err}")

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates the DataFrame is a time indexed pandas dataframe,
        or it has one column named as "date" or "datetime".


        Args:
            df: a time indexed pandas dataframe, or a pandas dataframe
            with one column named as "date" or "datetime"

        Raises:
            TypeError: raised when df is not a pandas dataframe
            ValueError: raised when there is no datetime index and no
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

        # create copy of df so it's internal to the instance
        df = df.copy()

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
            return self._set_col_as_datetime_index(df, "datetime")

        # lastly, use date as index
        if has_date:
            return self._set_col_as_datetime_index(df, "date")

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
