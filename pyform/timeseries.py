import logging
import pandas as pd

log = logging.getLogger(__name__)


class TimeSeries:
    def __init__(self, df: pd.DataFrame):

        df = self._validate_input(df)
        self.df = df

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
        """Validates the DataFrame is a time indexed pandas dataframe,
        or it has one column named as "date" or "datetime". 
        
        For a timeseries object, time index should be unique, meaning
        it would only accept a "wide" dataframe and not a "long" dataframe.

        Args:
            df: a time indexed pandas dataframe, or a pandas dataframe
            with one column named as "date" or "datetime"
        
        Raises:
            TypeError: raised when df is not a pandas dataframe

        Returns:
            pd.DataFrame: formatted dataframe, with datetime index and
            one column of data
        """

        # Check we are getting a pandas dataframe
        try:
            assert isinstance(df, pd.DataFrame)
        except AssertionError:
            raise TypeError(f"TimeSeries df argument must be a pandas DataFrame")

        # We are getting a pandas dataframe, and it is datetime indexed.
        # Return it.
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        # We are getting a pandas dataframe, but without datetime index.
        # See if one of the columns can be converted to the required datetime index.
        has_datetime = "datetime" in df
        has_date = "date" in df

        try:
            assert has_datetime or has_date
        except AssertionError:
            raise ValueError(
                "TimeSeries df argument without DatetimeIndex"
                "should have a 'date' or 'datetime' column."
            )

        # datetime column is preferred, as the name suggests it also has time in it,
        # which helps make the time series more precise
        if has_datetime:
            log.info("Using 'datetime' column as index.")
            try:
                df = df.set_index("datetime")
                df.index = pd.to_datetime(df.index)
                return df
            except Exception as err:
                if has_date:
                    # If dataframe also has date column, try date column before failing
                    log.warn(f"Error converting 'datetime' to index: {err}")
                    pass
                else:
                    raise ValueError(f"Error converting 'datetime' to index: {err}")

        # lastly, use date as index
        if has_date:
            log.info("Using 'date' column as index.")
            try:
                df = df.set_index("date")
                df.index = pd.to_datetime(df.index)
                return df
            except Exception as err:
                raise ValueError(f"Error converting 'date' to index: {err}")
