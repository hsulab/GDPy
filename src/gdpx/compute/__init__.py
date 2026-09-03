"""Deprecated compatibility namespace for :mod:`gdpx.execution.lifecycle`."""

import warnings

warnings.warn(
    "gdpx.compute is deprecated; import from gdpx.execution.lifecycle.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.execution.lifecycle import *  # noqa: E402,F401,F403
