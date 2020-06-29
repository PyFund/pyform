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

        # start and end date of the series
        self.start = min(self.series.index)
        self.end = max(self.series.index)

        # frequency of the series
        self.freq = self._infer_freq()

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
            ValueError: when column cannot be converted to datetime index

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

    @staticmethod
    def _freq_compare(freq1: str, freq2: str) -> bool:
        """Tests freq1 has a lower frequency than freq2.
        Lower frequencies cannot be converted to higher frequencies
        due to lower resolution.

        Args:
            freq1: frequency 1
            freq2: frequency 2

        Returns:
            bool: frequency 1 is lower than or euqal to frequency 2
        """

        freq = ["H", "D", "B", "W", "M", "Q", "Y"]
        freq = dict(zip(freq, [*range(0, len(freq))]))

        return freq[freq1] >= freq[freq2]

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates the DataFrame is a time indexed pandas dataframe,
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

    def reset(self):
        """Reset data to its initial state
        """

        self.series = self._series.copy()
        self.start = min(self.series.index)
        self.end = max(self.series.index)

    def _infer_freq(self) -> str:
        """Infer the frequency of the time series

        Raises:
            ValueError: when multiple frequencies are detected
            ValueError: when no frequency can be detected

        Returns:
            str: frequency of the time series
        """

        freq = set()
        max_check = min(len(self.series.index) - 10, 50)

        if max_check <= 10:
            inferred_freq = self.series.index.inferred_freq
            if inferred_freq is not None:
                freq.add(inferred_freq)
        else:
            # check head
            for i in range(0, max_check, 10):

                inferred_freq = self.series.index[i : (i + 10)].inferred_freq

                if inferred_freq is not None:

                    freq.add(inferred_freq)

            # check from tail
            for i in range(0, -max_check, -10):

                inferred_freq = self.series.index[(i - 11) : (i - 1)].inferred_freq

                if inferred_freq is not None:

                    freq.add(inferred_freq)

        if len(freq) == 0:
            raise ValueError("Cannot infer series frequency.")

        if len(freq) > 1:
            raise ValueError(f"Multiple series frequency detected: {freq}")

        return freq.pop()
