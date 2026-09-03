"""Deprecated compatibility namespace for :mod:`gdpx.modifiers.bias`."""

import warnings

warnings.warn(
    "gdpx.bias is deprecated; import from gdpx.modifiers.bias.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.modifiers.bias import *  # noqa: E402,F401,F403
