"""Deprecated compatibility namespace for :mod:`gdpx.structures.topology`."""

import warnings

warnings.warn(
    "gdpx.graph is deprecated; import from gdpx.structures.topology.",
    DeprecationWarning,
    stacklevel=2,
)
