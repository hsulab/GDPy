"""Shared configuration helpers for global-optimisation objectives."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Optional


DEFAULT_OBJECTIVE = {"target": "energy"}


def normalise_objective(
    objective: Optional[Mapping],
    supported_targets: set[str],
) -> dict:
    """Validate and canonicalise a search objective."""
    if objective is None:
        return copy.deepcopy(DEFAULT_OBJECTIVE)
    if not isinstance(objective, Mapping):
        raise TypeError("objective must be a mapping.")

    objective = copy.deepcopy(dict(objective))
    if "chempot" in objective:
        raise ValueError(
            "Legacy objective key 'chempot' is not supported; use "
            "'chemical_potentials'."
        )

    target = objective.get("target", "energy")
    if target not in supported_targets:
        choices = ", ".join(sorted(supported_targets))
        raise ValueError(
            f"objective.target {target!r} is not supported; choose one of: {choices}."
        )
    objective["target"] = target

    if target != "energy":
        chemical_potentials = objective.get("chemical_potentials")
        if not isinstance(chemical_potentials, Mapping) or not chemical_potentials:
            raise ValueError(
                "objective.chemical_potentials must be a non-empty mapping for "
                f"target {target!r}."
            )
        objective["chemical_potentials"] = copy.deepcopy(dict(chemical_potentials))

    return objective


def is_default_objective(objective: Mapping) -> bool:
    """Return whether an objective is the implicit energy default."""
    return dict(objective) == DEFAULT_OBJECTIVE


def reject_legacy_property(kwargs: dict) -> None:
    """Reject the former recipe-level scoring key with migration guidance."""
    if "property" in kwargs:
        raise ValueError(
            "Legacy search scoring key 'property' is not supported; use 'objective'."
        )
