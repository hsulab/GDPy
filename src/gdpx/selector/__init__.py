"""Deprecated compatibility namespace for :mod:`gdpx.analysis.selectors`."""

import warnings

warnings.warn(
    "gdpx.selector is deprecated; import from gdpx.analysis.selectors.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.analysis.selectors import *  # noqa: E402,F401,F403
