"""Deprecated compatibility namespace for :mod:`gdpx.workflow.nodes`."""

import warnings

warnings.warn(
    "gdpx.nodes is deprecated; import from gdpx.workflow.nodes.",
    DeprecationWarning,
    stacklevel=2,
)
