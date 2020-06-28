import pytest

from pyform.returnseries import ReturnSeries


def test_init():

    ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")


def test_to_week():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    assert returns.to_week().iloc[0, 0] == -0.07394800000000001


def test_to_month():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    returns.to_month().iloc[0, 0] == -0.07601189285351273


def test_to_quarter():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    returns.to_quarter().iloc[0, 0] == 0.42607684639593835


def test_to_year():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    returns.to_year().iloc[0, 0] == 0.42607684639593835

def test_to_week_arithmetic():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    assert returns.to_week("arithmetic").iloc[1, 0] == 0.05766

def test_to_week_continuous():

    returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
    assert returns.to_week("continuous").iloc[1, 0] == 0.05935475385633149

def test_to_period_wrong_method():

    with pytest.raises(ValueError):
        returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")
        returns.to_freq("W", "contnuuous") # typo in continuous should cause failure
