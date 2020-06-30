import pytest
import pandas as pd
from pyform.timeseries import TimeSeries
from pyform.util.freq import is_lower_freq, infer_freq


def test_freq_compare():

    assert is_lower_freq("W", "D")
    assert is_lower_freq("D", "D")
    assert not is_lower_freq("D", "W")


def test_infer_freq():

    ts = TimeSeries.read_csv("tests/unit/data/twitter.csv")
    assert infer_freq(ts) == "B"

    # The following should raise value error, since first 10 are daily, and
    # last 10 are monthly frequencies
    df = pd.DataFrame(
        data={
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-11",
                "2020-02-01",
                "2020-03-01",
                "2020-04-01",
                "2020-05-01",
                "2020-06-01",
                "2020-07-01",
                "2020-08-01",
                "2020-09-01",
                "2020-10-01",
                "2020-11-01",
                "2020-12-01",
            ],
            "returns": [*range(0, 22)],
        }
    )
    with pytest.raises(ValueError):
        infer_freq(TimeSeries(df))

    # The following should raise value error, as frequencies are mixed and
    # cannot be inferred
    df = pd.DataFrame(
        data={
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-06-01",
                "2020-07-01",
                "2020-08-01",
                "2020-09-01",
            ],
            "returns": [*range(0, 6)],
        }
    )
    with pytest.raises(ValueError):
        infer_freq(TimeSeries(df))
