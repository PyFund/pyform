from pyform.timeseries import TimeSeries

import datetime
import pytest
import pandas as pd


def test_validate_input_invalid_type():
    """Test when given a wrong type of input, TypeError is raised
    """

    with pytest.raises(TypeError):
        TimeSeries(0)


def test_validate_input_no_datetime():
    """Test when give a dataframe with no datetime index and no
    datetime columns, ValueError is raised
    """

    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    with pytest.raises(ValueError):
        TimeSeries(df)


def test_validate_input_use_datetime():
    """Validate when no datetime index is given, use datetime
    column as index if present
    """

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


def test_validate_input_use_date():
    """Validate when no datetime index is given, use date
    column as index if present
    """

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


def test_validate_bad_datetime():
    """Validate when no datetime index is given, value error is
    raised if datetime column cannot be converted to datetime index
    """

    df = pd.DataFrame(
        data={
            "datetime": ["20200101", "20200102", "20200103a"],  # last date is invalid
            "returns": [0.0, -0.1, 0.3],
        }
    )

    with pytest.raises(ValueError):
        TimeSeries(df)


def test_validate_correct_input():
    """Validate when no datetime index is given, use date
    column as index if present
    """

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
