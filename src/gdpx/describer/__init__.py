"""Deprecated compatibility namespace for :mod:`gdpx.analysis.descriptors`."""

import warnings

warnings.warn(
    "gdpx.describer is deprecated; import from gdpx.analysis.descriptors.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.analysis.descriptors import *  # noqa: E402,F401,F403
