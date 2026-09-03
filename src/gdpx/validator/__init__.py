"""Deprecated compatibility namespace for :mod:`gdpx.analysis.validators`."""

import warnings

warnings.warn(
    "gdpx.validator is deprecated; import from gdpx.analysis.validators.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.analysis.validators import *  # noqa: E402,F401,F403
