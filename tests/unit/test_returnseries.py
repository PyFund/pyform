import datetime
import pytest
import pandas as pd
from pyform import ReturnSeries, CashSeries

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

    with pytest.raises(ValueError):
        returns.to_period("H", "geometric")  # converting data to higher frequency


def test_add_bm():

    returns.add_bm(spy, "S&P 500")
    assert "S&P 500" in returns.benchmark

    returns.add_bm(qqq)
    assert "QQQ" in returns.benchmark


def test_add_rf():

    returns.add_rf(libor1m, "libor")
    assert "libor" in returns.risk_free

    returns.add_rf(libor1m)
    assert "LIBOR_1M" in returns.risk_free


def test_corr():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # no benchmark should raise ValueError
    with pytest.raises(ValueError):
        returns.get_corr()

    # with single benchmark
    returns.add_bm(spy)

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
            "used": 80,
        }
    )
    assert corr.equals(expected_output)

    # test multiple benchmarks
    returns.add_bm(qqq)
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
    total_return = returns.get_tot_ret()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "total return",
            "value": [-0.35300922502128473],
        }
    )
    assert total_return.equals(expected_output)

    # with single benchmark
    returns.add_bm(spy)
    total_return = returns.get_tot_ret()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "total return",
            "value": [-0.35300922502128473, 0.6935467657365115],
        }
    )
    assert total_return.equals(expected_output)

    # meta=True
    total_return = returns.get_tot_ret(meta=True)
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
    total_return = returns.get_tot_ret(include_bm=False)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "total return",
            "value": [-0.35300922502128473],
        }
    )
    assert total_return.equals(expected_output)

    # test multiple benchmarks
    returns.add_bm(qqq)
    total_return = returns.get_tot_ret()
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
    ann_return = returns.get_ann_ret()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized return",
            "value": [-0.06350385733729014],
        }
    )
    assert ann_return.equals(expected_output)

    ann_return = returns.get_ann_ret(method="arithmetic", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized return",
            "value": [0.0855005949876238],
            "method": "arithmetic",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_return.equals(expected_output)

    ann_return = returns.get_ann_ret(method="continuous")
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized return",
            "value": [0.0855005949876238],
        }
    )
    assert ann_return.equals(expected_output)

    # with single benchmark
    returns.add_bm(spy)
    ann_return = returns.get_ann_ret()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "annualized return",
            "value": [-0.06350385733729014, 0.08261818990205616],
        }
    )
    assert ann_return.equals(expected_output)

    # meta=True
    ann_return = returns.get_ann_ret(meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "annualized return",
            "value": [-0.06350385733729014, 0.08261818990205616],
            "method": "geometric",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_return.equals(expected_output)

    # has benchmark, but include_bm=False
    ann_return = returns.get_ann_ret(include_bm=False)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized return",
            "value": [-0.06350385733729014],
        }
    )
    assert ann_return.equals(expected_output)


def test_annualized_volatility():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # No benchmark
    ann_vol = returns.get_ann_vol()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized volatility",
            "value": [0.5199859200287252],
        }
    )
    assert ann_vol.equals(expected_output)

    # daily volatility
    ann_vol = returns.get_ann_vol(freq="D", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized volatility",
            "value": [0.545190844726424],
            "freq": "D",
            "method": "sample",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_vol.equals(expected_output)

    # population standard deviation
    ann_vol = returns.get_ann_vol(method="population", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized volatility",
            "value": [0.5167257880784241],
            "freq": "M",
            "method": "population",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_vol.equals(expected_output)

    # with single benchmark
    returns.add_bm(spy)
    ann_vol = returns.get_ann_vol()
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "annualized volatility",
            "value": [0.5199859200287252, 0.13606427329407125],
        }
    )
    assert ann_vol.equals(expected_output)

    # daily volatility
    ann_vol = returns.get_ann_vol(freq="D", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR", "SPY"],
            "field": "annualized volatility",
            "value": [0.545190844726424, 0.17497687303106887],
            "freq": "D",
            "method": "sample",
            "start": datetime.datetime.strptime("2013-11-07", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert ann_vol.equals(expected_output)

    # has benchmark, but include_bm=False
    ann_vol = returns.get_ann_vol(include_bm=False)
    expected_output = pd.DataFrame(
        data={
            "name": ["TWTR"],
            "field": "annualized volatility",
            "value": [0.5199859200287252],
        }
    )
    assert ann_vol.equals(expected_output)


def test_sharpe_ratio():

    returns = ReturnSeries.read_csv("tests/unit/data/spy_returns.csv")

    # No benchmark
    sharpe_ratio = returns.get_sharpe()
    expected_output = pd.DataFrame(
        data={"name": ["SPY"], "field": "sharpe ratio", "value": [0.5319128667616774]}
    )
    assert sharpe_ratio.equals(expected_output)

    # daily
    sharpe_ratio = returns.get_sharpe(freq="D", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["SPY"],
            "field": "sharpe ratio",
            "value": [0.39292358311061165],
            "freq": "D",
            "risk_free": "cash_0: 0.0%",
            "start": datetime.datetime.strptime("2003-04-01", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert sharpe_ratio.equals(expected_output)

    # use libor for risk free rate
    returns.add_rf(libor1m, "libor")
    sharpe_ratio = returns.get_sharpe(freq="D", risk_free="libor", meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["SPY"],
            "field": "sharpe ratio",
            "value": [0.3175248036195898],
            "freq": "D",
            "risk_free": "LIBOR_1M: 1.54%",
            "start": datetime.datetime.strptime("2003-04-01", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-19", "%Y-%m-%d"),
        }
    )
    assert sharpe_ratio.equals(expected_output)

    # with benchmark
    returns.add_bm(qqq)
    sharpe_ratio = returns.get_sharpe(meta=True)
    expected_output = pd.DataFrame(
        data={
            "name": ["SPY", "QQQ"],
            "field": "sharpe ratio",
            "value": [0.5319128667616774, 0.8028116328839393],
            "freq": "M",
            "risk_free": "cash_0: 0.0%",
            "start": datetime.datetime.strptime("2003-04-01", "%Y-%m-%d"),
            "end": datetime.datetime.strptime("2020-06-26", "%Y-%m-%d"),
        }
    )
    assert sharpe_ratio.equals(expected_output)

    # wrong key
    with pytest.raises(ValueError):
        returns.get_sharpe(risk_free="not-exist")

    # wrong type
    with pytest.raises(TypeError):
        returns.get_sharpe(risk_free=libor1m)


def test_rolling_tot_ret():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # No benchmark
    roll_tot_ret = returns.get_rolling_tot_ret()
    roll_twtr = roll_tot_ret["TWTR"]
    assert roll_twtr.index[0] == datetime.datetime.strptime("2016-10-31", "%Y-%m-%d")
    assert roll_twtr["TWTR"][0] == -0.6002237318946346

    # Daily, rolling 252 days
    roll_tot_ret = returns.get_rolling_tot_ret(window=252, freq="D")
    roll_twtr = roll_tot_ret["TWTR"]
    assert roll_twtr.index[0] == datetime.datetime.strptime("2014-11-06", "%Y-%m-%d")
    assert roll_twtr["TWTR"][0] == -0.09043080731969699

    returns.add_bm(spy)
    roll_tot_ret = returns.get_rolling_tot_ret()
    roll_twtr = roll_tot_ret["TWTR"]
    roll_spy = roll_tot_ret["SPY"]
    assert roll_twtr.index[0] == datetime.datetime.strptime("2016-10-31", "%Y-%m-%d")
    assert roll_twtr["TWTR"][0] == -0.6002237318946346
    assert roll_spy.index[0] == datetime.datetime.strptime("2016-10-31", "%Y-%m-%d")
    assert roll_spy["SPY"][0] == 0.1996927920869329


def test_rolling_ann_vol():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # No benchmark
    roll_ann_vol = returns.get_rolling_ann_vol()
    roll_twtr = roll_ann_vol["TWTR"]
    assert roll_twtr.index[0] == datetime.datetime.strptime("2016-10-31", "%Y-%m-%d")
    assert roll_twtr["TWTR"][0] == 0.5725024779205684

    # Daily, rolling 252 days
    roll_ann_vol = returns.get_rolling_ann_vol(window=252, freq="D")
    roll_twtr = roll_ann_vol["TWTR"]
    assert roll_twtr.index[0] == datetime.datetime.strptime("2014-11-06", "%Y-%m-%d")
    assert roll_twtr["TWTR"][0] == 0.6376491116934246

    returns.add_bm(spy)
    roll_ann_vol = returns.get_rolling_ann_vol()
    roll_twtr = roll_ann_vol["TWTR"]
    roll_spy = roll_ann_vol["SPY"]
    assert roll_twtr.index[0] == datetime.datetime.strptime("2016-10-31", "%Y-%m-%d")
    assert roll_twtr["TWTR"][0] == 0.5725024779205684
    assert roll_spy.index[0] == datetime.datetime.strptime("2016-10-31", "%Y-%m-%d")
    assert roll_spy["SPY"][0] == 0.10810183559733508


def test_index_series():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # No benchmark
    index_series = returns.get_index_series()
    index_twitter = index_series["TWTR"]
    assert index_twitter.index[-1] == datetime.datetime.strptime(
        "2020-06-30", "%Y-%m-%d"
    )
    assert index_twitter["TWTR"][-1] == -0.35300922502128296

    # Daily
    index_series = returns.get_index_series(freq="D")
    index_twitter = index_series["TWTR"]
    assert index_twitter.index[-1] == datetime.datetime.strptime(
        "2020-06-26", "%Y-%m-%d"
    )
    assert index_twitter["TWTR"][-1] == -0.35300922502128473

    returns.add_bm(spy)
    index_series = returns.get_index_series()
    index_twitter = index_series["TWTR"]
    index_spy = index_series["SPY"]
    assert index_twitter.index[-1] == datetime.datetime.strptime(
        "2020-06-30", "%Y-%m-%d"
    )
    assert index_spy.index[-1] == datetime.datetime.strptime("2020-06-30", "%Y-%m-%d")
    assert index_twitter["TWTR"][-1] == -0.35300922502128296
    assert index_spy["SPY"][-1] == 0.6935467657365093


def test_rolling_ann_return():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")

    # No benchmark
    roll_ann_ret = returns.get_rolling_ann_ret()
    roll_twtr = roll_ann_ret["TWTR"]
    assert roll_twtr.index[0] == datetime.datetime.strptime("2016-10-31", "%Y-%m-%d")
    assert roll_twtr["TWTR"][0] == -0.2633310984499173

    # Daily, rolling 252 days
    roll_ann_ret = returns.get_rolling_ann_ret(window=252, freq="D")
    roll_twtr = roll_ann_ret["TWTR"]
    assert roll_twtr.index[0] == datetime.datetime.strptime("2014-11-06", "%Y-%m-%d")
    assert roll_twtr["TWTR"][0] == -0.09048985526180642

    returns.add_bm(spy)
    roll_ann_ret = returns.get_rolling_ann_ret()
    roll_twtr = roll_ann_ret["TWTR"]
    roll_spy = roll_ann_ret["SPY"]
    assert roll_twtr.index[0] == datetime.datetime.strptime("2016-10-31", "%Y-%m-%d")
    assert roll_twtr["TWTR"][0] == -0.2633310984499173
    assert roll_spy.index[0] == datetime.datetime.strptime("2016-10-31", "%Y-%m-%d")
    assert roll_spy["SPY"][0] == 0.06256787890936222


def test_libor_fred():

    CashSeries.read_fred_libor_1m()
