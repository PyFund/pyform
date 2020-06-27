from pyform.timeseries import TimeSeries

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

    assert expected_output.equals(ts.df)


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

    assert expected_output.equals(ts.df)


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

    assert expected_output.equals(ts.df)


def test_init_from_csv():
    """Validate the read_csv clasmethod can initiate
    timeseries objects from csv
    """

    TimeSeries.read_csv("tests/unit/data/twitter.csv")