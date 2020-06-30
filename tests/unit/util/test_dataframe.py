import datetime
import pandas as pd
from pyform.util.dataframe import set_col_as_datetime_index


def test_set_col_as_datetime_index():

    df = pd.DataFrame(
        data={
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "returns": [*range(0, 3)],
        }
    )

    df = set_col_as_datetime_index(df, "date")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")
