"""Deprecated compatibility imports for MD execution helpers."""

import warnings

warnings.warn(
    "gdpx.computation.md.md_utils is deprecated; import from gdpx.execution.md.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.execution.md import force_temperature  # noqa: E402,F401

__all__ = ["force_temperature"]
