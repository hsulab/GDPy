"""Shared configuration and candidate scoring for global-optimisation objectives."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Optional

import numpy as np
from ase import Atoms

DEFAULT_OBJECTIVE = {"target": "energy"}


def evaluate_candidate(
    atoms: Atoms,
    objective_target: str,
    chemical_potentials: Optional[Mapping] = None,
) -> None:
    """Store the minimised objective and its negation as maximised fitness.

    Cohesive energy subtracts elemental references for every atom. Formation
    energy subtracts references weighted by the caller's fragment/species
    identity counts. Neither objective is normalised per atom here.
    """
    pairs = atoms.info["key_value_pairs"]
    assert pairs.get("raw_score") is None, "candidate already has raw_score before evaluation"

    if objective_target == "energy":
        target = atoms.get_potential_energy()
    elif objective_target in {"cohesive_energy", "formation_energy"}:
        assert chemical_potentials is not None, (
            f"chemical_potentials must not be None for {objective_target}."
        )
        if objective_target == "cohesive_energy":
            references = [chemical_potentials[symbol] for symbol in atoms.get_chemical_symbols()]
        else:
            identity_stats = atoms.info.get("identity_stats")
            assert identity_stats is not None, (
                "Fail to compute `formation_energy` as no `identity_stats` is found in atoms.info."
            )
            references = [chemical_potentials[name] * count for name, count in identity_stats.items()]
        target = atoms.get_potential_energy() - np.sum(references)
    elif objective_target == "reaction_energy":
        raise NotImplementedError("reaction_energy is not implemented.")
    else:
        raise RuntimeError(f"Unknown target {objective_target}...")

    pairs["target"] = target
    pairs["raw_score"] = -target


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
