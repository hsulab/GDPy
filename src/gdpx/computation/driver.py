"""Deprecated compatibility imports for the execution driver core."""

import warnings

warnings.warn(
    "gdpx.computation.driver is deprecated; import driver primitives from gdpx.execution.driver.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.execution.driver import (  # noqa: E402,F401
    EARLYSTOP_KEY,
    BaseDriver,
    Controller,
    DriverSetting,
    check_constraint_consistency,
)

__all__ = [
    "EARLYSTOP_KEY",
    "BaseDriver",
    "Controller",
    "DriverSetting",
    "check_constraint_consistency",
]
