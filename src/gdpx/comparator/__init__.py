"""Deprecated compatibility namespace for :mod:`gdpx.analysis.comparators`."""

import warnings

warnings.warn(
    "gdpx.comparator is deprecated; import from gdpx.analysis.comparators.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.analysis.comparators import *  # noqa: E402,F401,F403
