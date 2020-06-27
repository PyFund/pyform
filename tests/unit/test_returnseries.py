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
