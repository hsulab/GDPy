"""Deprecated compatibility namespace for :mod:`gdpx.data.loaders`."""

import warnings

warnings.warn(
    "gdpx.dataloader is deprecated; import from gdpx.data.loaders.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.data.loaders import *  # noqa: E402,F401,F403
