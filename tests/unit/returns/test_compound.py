import pytest
import pandas as pd
from pyform.returns.compound import (
    compound_geometric,
    compound_arithmetic,
    compound_continuous,
    compound,
    cumseries_geometric,
    cumseries_arithmetic,
    cumseries_continuous,
    cumseries,
)


def test_compound():

    assert compound("geometric") == compound_geometric
    assert compound("arithmetic") == compound_arithmetic
    assert compound("continuous") == compound_continuous

    with pytest.raises(ValueError):
        compound("contnuuous")  # typo in continuous should cause failure


def test_compound_methods():

    returns = pd.Series(
        [0.030011999999999997, -0.02331, 0.016706000000000002, 0.049061, -0.015887]
    )

    assert compound_geometric(returns) == 0.055942142480424284
    assert compound_arithmetic(returns) == 0.05658200000000001
    assert compound_continuous(returns) == 0.05821338474015869


def test_cumseries():

    assert cumseries("geometric") == cumseries_geometric
    assert cumseries("arithmetic") == cumseries_arithmetic
    assert cumseries("continuous") == cumseries_continuous

    with pytest.raises(ValueError):
        cumseries("contnuuous")  # typo in continuous should cause failure


def test_cumseries_methods():

    returns = pd.Series(
        [0.030011999999999997, -0.02331, 0.016706000000000002, 0.049061, -0.015887]
    )

    assert cumseries_geometric(returns).iloc[-1] == 0.055942142480424284
    assert cumseries_arithmetic(returns).iloc[-1] == 0.05658200000000001
    assert cumseries_continuous(returns).iloc[-1] == 0.05821338474015869
