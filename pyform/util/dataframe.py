import pandas as pd


def set_col_as_datetime_index(df: pd.DataFrame, col: str) -> pd.DataFrame:
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

    try:
        df = df.set_index(col)
        df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"
        return df
    except Exception as err:
        raise ValueError(f"Error converting '{col}' to index: {err}")
