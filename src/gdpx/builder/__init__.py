"""Deprecated compatibility namespace for :mod:`gdpx.structures.builders`."""

import warnings

warnings.warn(
    "gdpx.builder is deprecated; import from gdpx.structures.builders.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.structures.builders import *  # noqa: E402,F401,F403
