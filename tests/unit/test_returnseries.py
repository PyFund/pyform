import datetime
import pytest
import pandas as pd
from pyform.returnseries import ReturnSeries, CashSeries

returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
spy = ReturnSeries.read_csv("tests/unit/data/spy_returns.csv")
qqq = ReturnSeries.read_csv("tests/unit/data/qqq_returns.csv")
libor1m = ReturnSeries.read_csv("tests/unit/data/libor1m_returns.csv")


def test_init():

    df = pd.read_csv("tests/unit/data/twitter_returns.csv")
    ReturnSeries(df, "Twitter")


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

    returns.add_benchmark(spy, "S&P 500")
    assert "S&P 500" in returns.benchmark

    returns.add_benchmark(qqq)
    assert "QQQ" in returns.benchmark


def test_add_risk_free():

    returns.add_risk_free(libor1m, "libor")
    assert "libor" in returns.risk_free

    returns.add_risk_free(libor1m)
    assert "LIBOR_1M" in returns.risk_free


def test_corr():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # no benchmark should raise ValueError
    with pytest.raises(ValueError):
        returns.get_corr()

    # with single benchmark
    returns.add_benchmark(spy)

    corr = returns.get_corr()
    expected_output = pd.DataFrame(
        data={"name": ["SPY"], "field": "correlation", "value": [0.21224719919904408]}
    )
    assert corr.equals(expected_output)

    corr = returns.get_corr(meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["SPY"],
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
            "name": ["SPY", "QQQ"],
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

    # with single benchmark
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

    # with single benchmark
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


def test_annualized_volatility():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # No benchmark
    ann_vol = returns.get_annualized_volatility()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized volatility",
            "value": [0.5200932110481337],
        }
    )
    assert ann_vol.equals(expected_output)

    # daily volatility
    ann_vol = returns.get_annualized_volatility(freq="D", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized volatility",
            "value": [0.545298443138832],
            "freq": "D",
            "method": "sample",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_vol.equals(expected_output)

    # population standard deviation
    ann_vol = returns.get_annualized_volatility(method="population", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized volatility",
            "value": [0.5168324064202331],
            "freq": "M",
            "method": "population",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_vol.equals(expected_output)

    # with single benchmark
    returns.add_benchmark(spy)
    ann_vol = returns.get_annualized_volatility()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "annualized volatility",
            "value": [0.5200932110481337, 0.13609234804383752],
        }
    )
    assert ann_vol.equals(expected_output)

    # daily volatility
    ann_vol = returns.get_annualized_volatility(freq="D", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "annualized volatility",
            "value": [0.545298443138832, 0.17501475527479404],
            "freq": "D",
            "method": "sample",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_vol.equals(expected_output)

    # has benchmark, but include_bm=False
    ann_vol = returns.get_annualized_volatility(include_bm=False)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized volatility",
            "value": [0.5200932110481337],
        }
    )
    assert ann_vol.equals(expected_output)


def test_sharpe_ratio():

    returns = ReturnSeries.read_csv("tests/unit/data/spy_returns.csv")

    # No benchmark
    sharpe_ratio = returns.get_sharpe_ratio()
    expected_output = pd.DataFrame(
        data={"name": ["SPY"], "field": "sharpe ratio", "value": [0.5319582050650019]}
    )
    assert sharpe_ratio.equals(expected_output)

    # daily
    sharpe_ratio = returns.get_sharpe_ratio(freq="D", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["SPY"],
            "field": "sharpe ratio",
            "value": [0.392952489244965],
            "freq": "D",
            "risk_free": "cash_0: 0.0%",
            "start": datetime.datetime.strptime("2003-04-01", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert sharpe_ratio.equals(expected_output)

    # use libor for risk free rate
    returns.add_risk_free(libor1m, "libor")
    sharpe_ratio = returns.get_sharpe_ratio(freq="D", risk_free="libor", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["SPY"],
            "field": "sharpe ratio",
            "value": [0.31751335834676475],
            "freq": "D",
            "risk_free": "LIBOR_1M: 1.54%",
            "start": datetime.datetime.strptime("2003-04-01", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-19", "%Y-%m-%d"),
        }
    )
    assert sharpe_ratio.equals(expected_output)

    # with benchmark
    returns.add_benchmark(qqq)
    sharpe_ratio = returns.get_sharpe_ratio(meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["SPY", "QQQ"],
            "field": "sharpe ratio",
            "value": [0.5319582050650019, 0.8028838684875361],
            "freq": "M",
            "risk_free": "cash_0: 0.0%",
            "start": datetime.datetime.strptime("2003-04-01", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert sharpe_ratio.equals(expected_output)

    # wrong key
    with pytest.raises(ValueError):
        returns.get_sharpe_ratio(risk_free="not-exist")

    # wrong type
    with pytest.raises(TypeError):
        returns.get_sharpe_ratio(risk_free=libor1m)


def test_libor_fred():

    CashSeries.read_fred_libor_1m()
