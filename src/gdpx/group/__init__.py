"""Deprecated compatibility namespace for :mod:`gdpx.structures.groups`."""

import warnings

warnings.warn(
    "gdpx.group is deprecated; import from gdpx.structures.groups.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.structures.groups import evaluate_constraint_expression, evaluate_group_expression  # noqa: E402,F401

__all__ = ["evaluate_group_expression", "evaluate_constraint_expression"]
