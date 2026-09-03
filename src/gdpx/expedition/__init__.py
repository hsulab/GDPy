"""Deprecated compatibility namespace for :mod:`gdpx.exploration`."""

import warnings

warnings.warn(
    "gdpx.expedition is deprecated; import from gdpx.exploration.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.exploration import *  # noqa: E402,F401,F403
