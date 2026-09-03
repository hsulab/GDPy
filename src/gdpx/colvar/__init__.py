"""Deprecated compatibility namespace for :mod:`gdpx.modifiers.collective_variables`."""

import warnings

warnings.warn(
    "gdpx.colvar is deprecated; import from gdpx.modifiers.collective_variables.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.modifiers.collective_variables import *  # noqa: E402,F401,F403
