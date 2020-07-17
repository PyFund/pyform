# flake8: noqa

# Versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from pyform.returnseries import ReturnSeries, CashSeries
