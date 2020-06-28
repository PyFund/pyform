import datetime
import pytest
import pandas as pd
from pyform.returnseries import ReturnSeries

returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
spy = ReturnSeries.read_csv("tests/unit/data/spy_returns.csv")
qqq = ReturnSeries.read_csv("tests/unit/data/qqq_returns.csv")


def test_to_period():

    assert returns.to_week().iloc[1, 0] == 0.055942142480424284
    assert returns.to_month().iloc[1, 0] == 0.5311520760874386
    assert returns.to_quarter().iloc[1, 0] == -0.2667730077753935
    assert returns.to_year().iloc[1, 0] == -0.4364528678695403
    assert returns.to_week("arithmetic").iloc[1, 0] == 0.05658200000000001
    assert returns.to_week("continuous").iloc[1, 0] == 0.05821338474015869
    with pytest.raises(ValueError):
        returns.to_period("W", "contnuuous")  # typo in continuous should cause failure
    with pytest.raises(ValueError):
        returns.to_period("H", "geometric")  # converting data to higher frequency


def test_add_benchmark():

    returns.add_benchmark(spy, "SPY")
    assert "SPY" in returns.benchmark

    returns.add_benchmark(qqq)
    assert "QQQ" in returns.benchmark


def test_corr():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # no benchmark should raise ValueError
    with pytest.raises(ValueError):
        returns.get_corr()

    # Single benchmark
    returns.add_benchmark(spy)

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
            "freq": "M",
            "method": "pearson",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
            "total": 80,
            "skipped": 0,
        }
    )
    assert corr.equals(expected_output)

    # test multiple benchmarks
    returns.add_benchmark(qqq)
    corr = returns.get_corr()
    expected_output = pd.DataFrame(
        data={
            "benchmark": ["SPY", "QQQ"],
            "field": "correlation",
            "value": [0.21224719919904408, 0.27249109347246325],
        }
    )
    assert corr.equals(expected_output)


def test_total_return():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # No benchmark
    total_return = returns.get_total_return()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "total return",
            "value": [-0.35300922502128473],
        }
    )
    assert total_return.equals(expected_output)

    # Single benchmark
    returns.add_benchmark(spy)
    total_return = returns.get_total_return()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "total return",
            "value": [-0.35300922502128473, 0.6935467657365115],
        }
    )
    assert total_return.equals(expected_output)

    # meta=True
    total_return = returns.get_total_return(meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "total return",
            "value": [-0.35300922502128473, 0.6935467657365115],
            "method": "geometric",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert total_return.equals(expected_output)

    # has benchmark, but include_bm=False
    total_return = returns.get_total_return(include_bm=False)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "total return",
            "value": [-0.35300922502128473],
        }
    )
    assert total_return.equals(expected_output)

    # test multiple benchmarks
    returns.add_benchmark(qqq)
    total_return = returns.get_total_return()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY", "QQQ"],
            "field": "total return",
            "value": [-0.35300922502128473, 0.6935467657365115, 1.894217403555647],
        }
    )
    assert total_return.equals(expected_output)


def test_annualized_return():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # No benchmark
    ann_return = returns.get_annualized_return()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized return",
            "value": [-0.06352921539090761],
        }
    )
    assert ann_return.equals(expected_output)

    ann_return = returns.get_annualized_return(method="arithmetic", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized return",
            "value": [0.08553588206768473],
            "method": "arithmetic",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_return.equals(expected_output)

    ann_return = returns.get_annualized_return(method="continuous")
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized return",
            "value": [0.08553588206768473],
        }
    )
    assert ann_return.equals(expected_output)

    # Single benchmark
    returns.add_benchmark(spy)
    ann_return = returns.get_annualized_return()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "annualized return",
            "value": [-0.06352921539090761, 0.08265365923419554],
        }
    )
    assert ann_return.equals(expected_output)

    # meta=True
    ann_return = returns.get_annualized_return(meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "annualized return",
            "value": [-0.06352921539090761, 0.08265365923419554],
            "method": "geometric",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_return.equals(expected_output)

    # has benchmark, but include_bm=False
    ann_return = returns.get_annualized_return(include_bm=False)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized return",
            "value": [-0.06352921539090761],
        }
    )
    assert ann_return.equals(expected_output)