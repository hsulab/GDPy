"""Deprecated compatibility import for path-execution primitives."""

import warnings

warnings.warn(
    "gdpx.reactor.reactor is deprecated; import BaseReactor from gdpx.execution.reactor.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.execution.reactor import BaseReactor  # noqa: E402,F401

__all__ = ["BaseReactor"]
