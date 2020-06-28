import datetime
import pytest
import pandas as pd
from pyform.returnseries import ReturnSeries


def test_init():

    ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")


def test_to_week():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    assert returns.to_week().iloc[1, 0] == 0.055942142480424284


def test_to_month():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    assert returns.to_month().iloc[1, 0] == 0.5311520760874386


def test_to_quarter():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    assert returns.to_quarter().iloc[1, 0] == -0.2667730077753935


def test_to_year():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    assert returns.to_year().iloc[1, 0] == -0.4364528678695403


def test_to_week_arithmetic():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    assert returns.to_week("arithmetic").iloc[1, 0] == 0.05658200000000001


def test_to_week_continuous():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    assert returns.to_week("continuous").iloc[1, 0] == 0.05821338474015869


def test_to_period_wrong_method():

    with pytest.raises(ValueError):
        returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
        returns.to_freq("W", "contnuuous")  # typo in continuous should cause failure


def test_add_benchmark():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    benchmark = ReturnSeries.read_csv("tests/unit/data/spy_returns.csv")
    returns.add_benchmark(benchmark, "SPY")

    assert "SPY" in returns.benchmark


def test_add_benchmark_no_name():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    benchmark = ReturnSeries.read_csv("tests/unit/data/spy_returns.csv")
    returns.add_benchmark(benchmark)

    corr = returns.get_corr()
    expected_output = pd.DataFrame(
        data={
            "benchmark": ["SPY"],
            "field": "correlation",
            "value": [0.21224719919904408],
        }
    )
    assert corr.equals(expected_output)

    corr = returns.get_corr(meta=True)
    expected_output = pd.DataFrame(
        data={
            "benchmark": ["SPY"],
            "field": "correlation",
            "value": [0.21224719919904408],
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
            "total": 80,
            "skipped": 0,
        }
    )
    assert corr.equals(expected_output)
