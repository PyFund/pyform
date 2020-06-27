from pyform.timeseries import TimeSeries

import pytest
import pandas as pd

def test_validate_input_invalid_type():
    """Test when given a wrong type of input, TypeError is raised
    """

    with pytest.raises(TypeError):
        TimeSeries(0)