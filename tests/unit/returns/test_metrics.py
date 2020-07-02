from pyform.returns.metrics import calc_ann_vol
from pyform.returnseries import ReturnSeries

returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")


def test_calc_ann_vol():

    assert calc_ann_vol(returns.series, "population") == 0.5450275894915236
    assert (
        calc_ann_vol(returns.series, "population", samples_per_year=252)
        == 0.5454208266167264
    )
