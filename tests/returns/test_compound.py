import pytest
import pandas as pd
from pyform.returns.compound import (
    compound_geometric,
    compound_arithmetic,
    compound_continuous,
    compound,
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
