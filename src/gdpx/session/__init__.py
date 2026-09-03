"""Deprecated compatibility namespace for :mod:`gdpx.workflow.session`."""

import warnings

warnings.warn(
    "gdpx.session is deprecated; import from gdpx.workflow.session.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.workflow.session import *  # noqa: E402,F401,F403
