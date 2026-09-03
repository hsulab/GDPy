"""Deprecated compatibility namespace for :mod:`gdpx.execution.workers`."""

import warnings

warnings.warn(
    "gdpx.worker is deprecated; import from gdpx.execution.workers.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.execution.workers import *  # noqa: E402,F401,F403
