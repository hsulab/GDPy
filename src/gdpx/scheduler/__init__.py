"""Deprecated compatibility namespace for :mod:`gdpx.execution.schedulers`."""

import warnings

warnings.warn(
    "gdpx.scheduler is deprecated; import from gdpx.execution.schedulers.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.execution.schedulers import *  # noqa: E402,F401,F403
