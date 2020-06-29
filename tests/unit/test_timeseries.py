from pyform.timeseries import TimeSeries

import datetime
import pytest
import pandas as pd


def test_validate_input():

    # Test when given a wrong type of input, TypeError is raised
    with pytest.raises(TypeError):
        TimeSeries(0)

    # Test when give a dataframe with no datetime index and no
    # datetime columns, ValueError is raised
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    with pytest.raises(ValueError):
        TimeSeries(df)

    # Validate when no datetime index is given, use datetime
    # column as index if present
    df = pd.DataFrame(
        data={
            "datetime": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "returns": [0.0, -0.1, 0.3],
        }
    )
    expected_output = df.copy()
    expected_output = expected_output.set_index("datetime")
    expected_output.index = pd.to_datetime(expected_output.index)
    ts = TimeSeries(df)
    assert expected_output.equals(ts.series)

    # Validate when no datetime index is given, use date
    # column as index if present
    df = pd.DataFrame(
        data={
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "returns": [0.0, -0.1, 0.3],
        }
    )
    expected_output = df.copy()
    expected_output = expected_output.set_index("date")
    expected_output.index = pd.to_datetime(expected_output.index)
    expected_output.index.name = "datetime"
    ts = TimeSeries(df)
    assert expected_output.equals(ts.series)

    # Validate when no datetime index is given, value error is
    # raised if datetime column cannot be converted to datetime index
    df = pd.DataFrame(
        data={
            "datetime": ["20200101", "20200102", "20200103a"],  # last date is invalid
            "returns": [0.0, -0.1, 0.3],
        }
    )
    with pytest.raises(ValueError):
        TimeSeries(df)

    # Validate timeseries can handle correct input
    df = pd.DataFrame(
        data={
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "returns": [0.0, -0.1, 0.3],
        },
    )
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    expected_output = df.copy()
    expected_output.index.name = "datetime"
    ts = TimeSeries(df)
    assert expected_output.equals(ts.series)


def test_init_from_csv():
    """Validate the read_csv clasmethod can initiate
    timeseries objects from csv
    """

    ts = TimeSeries.read_csv("tests/unit/data/twitter.csv")
    assert ts.series.iloc[0, 0] == 44.9


def test_init_from_excel():
    """Validate the read_excel clasmethod can initiate
    timeseries objects from excel
    """

    # TODO: update this test once we implement this method
    TimeSeries.read_excel("tests/unit/data/twitter.xlsx")


def test_init_from_db():
    """Validate the read_db clasmethod can initiate
    timeseries objects from database query
    """

    # TODO: update this test once we implement this method
    TimeSeries.read_db("SELECT * FROM returns")


def test_set_daterange():

    ts = TimeSeries.read_csv("tests/unit/data/twitter.csv")

    ts.set_daterange("2020-01-01", "2020-01-31")
    assert ts.start == datetime.datetime.strptime("2020-01-02", "%Y-%m-%d")
    assert ts.series.index[0] == datetime.datetime.strptime("2020-01-02", "%Y-%m-%d")
    assert ts.end == datetime.datetime.strptime("2020-01-31", "%Y-%m-%d")
    assert ts.series.index[-1] == datetime.datetime.strptime("2020-01-31", "%Y-%m-%d")

    ts.set_daterange(start="2020-01-01")
    assert ts.start == datetime.datetime.strptime("2020-01-02", "%Y-%m-%d")
    assert ts.series.index[0] == datetime.datetime.strptime("2020-01-02", "%Y-%m-%d")
    assert ts.end == datetime.datetime.strptime("2020-06-26", "%Y-%m-%d")
    assert ts.series.index[-1] == datetime.datetime.strptime("2020-06-26", "%Y-%m-%d")

    ts.set_daterange(end="2020-06-26")
    assert ts.start == datetime.datetime.strptime("2013-11-07", "%Y-%m-%d")
    assert ts.series.index[0] == datetime.datetime.strptime("2013-11-07", "%Y-%m-%d")
    assert ts.end == datetime.datetime.strptime("2020-06-26", "%Y-%m-%d")
    assert ts.series.index[-1] == datetime.datetime.strptime("2020-06-26", "%Y-%m-%d")


def test_reset_daterange():

    ts = TimeSeries.read_csv("tests/unit/data/twitter.csv")

    ts.set_daterange("2020-01-01", "2020-01-31")
    ts.reset()
    assert ts.start == datetime.datetime.strptime("2013-11-07", "%Y-%m-%d")
    assert ts.series.index[0] == datetime.datetime.strptime("2013-11-07", "%Y-%m-%d")
    assert ts.end == datetime.datetime.strptime("2020-06-26", "%Y-%m-%d")
    assert ts.series.index[-1] == datetime.datetime.strptime("2020-06-26", "%Y-%m-%d")


def test_infer_freq():

    ts = TimeSeries.read_csv("tests/unit/data/twitter.csv")
    assert ts.freq == "B"

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
        TimeSeries(df)

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
        TimeSeries(df)


def test_freq_compare():

    assert TimeSeries._freq_compare("W", "D")
    assert not TimeSeries._freq_compare("D", "W")
