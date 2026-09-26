from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii


SUPPORTED_SCOPES = {"inter_particle", "all_atoms"}


@dataclass(frozen=True)
class DistanceWindow:
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ContactCountRestraint:
    pair: tuple[int, int]
    distance: DistanceWindow
    minimum: Optional[int]
    maximum: Optional[int]
    scope: str = "inter_particle"


@dataclass(frozen=True)
class CoordinationRestraint:
    center: int
    neighbor: int
    distance: DistanceWindow
    coordination_minimum: Optional[int]
    coordination_maximum: Optional[int]
    matching_centers: Union[str, tuple[Optional[int], Optional[int]]]
    scope: str = "inter_particle"


ParsedRestraint = Union[ContactCountRestraint, CoordinationRestraint]


def _check_count(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"`{name}` must be a non-negative integer, but got {value!r}.")
    return value


def _parse_bounds(
    value: Mapping[str, Any], name: str
) -> tuple[Optional[int], Optional[int]]:
    unknown = set(value) - {"min", "max"}
    if unknown:
        raise ValueError(f"Unknown fields in `{name}`: {sorted(unknown)}.")
    minimum = _check_count(value.get("min"), f"{name}.min")
    maximum = _check_count(value.get("max"), f"{name}.max")
    if minimum is None and maximum is None:
        raise ValueError(f"`{name}` must define `min` or `max`.")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"`{name}.min` cannot be greater than `{name}.max`.")
    return minimum, maximum


def _atomic_number(symbol: Any, name: str) -> int:
    if not isinstance(symbol, str) or symbol not in atomic_numbers:
        raise ValueError(f"`{name}` must be a valid chemical symbol, but got {symbol!r}.")
    return atomic_numbers[symbol]


def _parse_scope(value: Any) -> str:
    if value not in SUPPORTED_SCOPES:
        raise ValueError(
            f"`scope` must be one of {sorted(SUPPORTED_SCOPES)}, but got {value!r}."
        )
    return value


def _parse_distance(
    value: Any,
    first: int,
    second: int,
    covalent_ratio: tuple[float, float],
) -> DistanceWindow:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("`distance` must be a mapping with optional `min` and `max` fields.")
    unknown = set(value) - {"min", "max"}
    if unknown:
        raise ValueError(f"Unknown fields in `distance`: {sorted(unknown)}.")

    reference = float(covalent_radii[first] + covalent_radii[second])
    minimum = value.get("min", covalent_ratio[0] * reference)
    maximum = value.get("max", covalent_ratio[1] * reference)
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise ValueError(f"`distance.min` must be a finite number, but got {minimum!r}.")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise ValueError(f"`distance.max` must be a finite number, but got {maximum!r}.")
    minimum, maximum = float(minimum), float(maximum)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError(f"`distance.min` must be finite and non-negative, but got {minimum!r}.")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"`distance.max` must be finite and positive, but got {maximum!r}.")
    if minimum >= maximum:
        raise ValueError("`distance.min` must be smaller than `distance.max`.")
    return DistanceWindow(minimum=minimum, maximum=maximum)


def parse_restraints(
    restraints: Optional[Sequence[Mapping[str, Any]]],
    covalent_ratio=(0.8, 2.0),
) -> list[ParsedRestraint]:
    """Validate and canonicalize geometric restraint configuration."""
    if restraints is None:
        return []
    if isinstance(restraints, (str, bytes)) or not isinstance(restraints, Sequence):
        raise ValueError("`restraints` must be a sequence of mappings.")
    if len(covalent_ratio) != 2:
        raise ValueError("`covalent_ratio` must contain exactly two values.")
    covalent_ratio = (float(covalent_ratio[0]), float(covalent_ratio[1]))
    if covalent_ratio[0] < 0.0 or covalent_ratio[0] >= covalent_ratio[1]:
        raise ValueError("`covalent_ratio` must satisfy 0 <= min < max.")

    parsed: list[ParsedRestraint] = []
    for index, config in enumerate(restraints):
        if not isinstance(config, Mapping):
            raise ValueError(f"`restraints[{index}]` must be a mapping.")
        restraint_type = config.get("type")
        scope = _parse_scope(config.get("scope", "inter_particle"))

        if restraint_type == "contact_count":
            unknown = set(config) - {"type", "pair", "min", "max", "distance", "scope"}
            if unknown:
                raise ValueError(f"Unknown fields in contact-count restraint: {sorted(unknown)}.")
            pair = config.get("pair")
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError("`pair` must contain exactly two chemical symbols.")
            first = _atomic_number(pair[0], "pair[0]")
            second = _atomic_number(pair[1], "pair[1]")
            minimum = _check_count(config.get("min"), "min")
            maximum = _check_count(config.get("max"), "max")
            if minimum is None and maximum is None:
                raise ValueError("A contact-count restraint must define `min` or `max`.")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError("Contact `min` cannot be greater than `max`.")
            parsed.append(
                ContactCountRestraint(
                    pair=tuple(sorted((first, second))),
                    distance=_parse_distance(config.get("distance"), first, second, covalent_ratio),
                    minimum=minimum,
                    maximum=maximum,
                    scope=scope,
                )
            )
        elif restraint_type == "coordination":
            unknown = set(config) - {
                "type",
                "center",
                "neighbor",
                "coordination",
                "matching_centers",
                "distance",
                "scope",
            }
            if unknown:
                raise ValueError(f"Unknown fields in coordination restraint: {sorted(unknown)}.")
            center = _atomic_number(config.get("center"), "center")
            neighbor = _atomic_number(config.get("neighbor"), "neighbor")
            coordination = config.get("coordination")
            if not isinstance(coordination, Mapping):
                raise ValueError("`coordination` must be a mapping with `min` and/or `max`.")
            coordination_minimum, coordination_maximum = _parse_bounds(coordination, "coordination")

            matching_centers = config.get("matching_centers")
            if matching_centers == "all":
                parsed_matching_centers: Union[str, tuple[Optional[int], Optional[int]]] = "all"
            elif isinstance(matching_centers, Mapping):
                parsed_matching_centers = _parse_bounds(matching_centers, "matching_centers")
            else:
                raise ValueError("`matching_centers` must be `all` or a min/max mapping.")

            parsed.append(
                CoordinationRestraint(
                    center=center,
                    neighbor=neighbor,
                    distance=_parse_distance(config.get("distance"), center, neighbor, covalent_ratio),
                    coordination_minimum=coordination_minimum,
                    coordination_maximum=coordination_maximum,
                    matching_centers=parsed_matching_centers,
                    scope=scope,
                )
            )
        else:
            raise ValueError(
                f"Unknown restraint type {restraint_type!r}; supported types are "
                "`contact_count` and `coordination`."
            )
    return parsed


def _uses_pair(restraint: ParsedRestraint, first: int, second: int) -> bool:
    pair = tuple(sorted((first, second)))
    if isinstance(restraint, ContactCountRestraint):
        return restraint.pair == pair
    return tuple(sorted((restraint.center, restraint.neighbor))) == pair


def _in_scope(restraint: ParsedRestraint, tags: Optional[np.ndarray], i: int, j: int) -> bool:
    if restraint.scope == "all_atoms":
        return True
    assert tags is not None
    return tags[i] != tags[j]


def validate_restraint_tags(atoms: Atoms, restraints: list[ParsedRestraint]) -> None:
    if any(restraint.scope == "inter_particle" for restraint in restraints) and not atoms.has("tags"):
        raise ValueError(
            "`inter_particle` restraints require an ASE `tags` array. Assign one tag per particle "
            "or use `scope: all_atoms`."
        )


def minimum_distance_for_pair(
    i: int,
    j: int,
    atomic_numbers_: np.ndarray,
    tags: Optional[np.ndarray],
    restraints: list[ParsedRestraint],
    default: float,
) -> float:
    """Return the effective minimum, allowing matching restraints to override the default."""
    matching = [
        restraint.distance.minimum
        for restraint in restraints
        if _uses_pair(restraint, int(atomic_numbers_[i]), int(atomic_numbers_[j]))
        and _in_scope(restraint, tags, i, j)
    ]
    return max(matching) if matching else default


def evaluate_restraints(
    atoms: Atoms,
    restraints: list[ParsedRestraint],
    *,
    complete: bool = True,
) -> bool:
    """Return whether an atomic structure satisfies all configured restraints.

    With ``complete=False``, only conditions that cannot recover after adding more
    atoms are checked: minimum distances and contact-count maxima.
    """
    if not restraints:
        return True
    validate_restraint_tags(atoms, restraints)
    numbers = atoms.get_atomic_numbers()
    tags = atoms.get_tags() if atoms.has("tags") else None
    distances = atoms.get_all_distances(mic=True)

    for restraint in restraints:
        contacts: list[tuple[int, int]] = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if not _uses_pair(restraint, int(numbers[i]), int(numbers[j])):
                    continue
                if not _in_scope(restraint, tags, i, j):
                    continue
                distance = float(distances[i, j])
                if distance < restraint.distance.minimum:
                    return False
                if distance < restraint.distance.maximum:
                    contacts.append((i, j))

        if isinstance(restraint, ContactCountRestraint):
            num_contacts = len(contacts)
            if restraint.maximum is not None and num_contacts > restraint.maximum:
                return False
            if complete and restraint.minimum is not None and num_contacts < restraint.minimum:
                return False
            continue

        if not complete:
            continue
        coordination = {i: 0 for i, number in enumerate(numbers) if number == restraint.center}
        for i, j in contacts:
            if numbers[i] == restraint.center and numbers[j] == restraint.neighbor:
                coordination[i] += 1
            if numbers[j] == restraint.center and numbers[i] == restraint.neighbor:
                coordination[j] += 1

        def matches(value: int) -> bool:
            if restraint.coordination_minimum is not None and value < restraint.coordination_minimum:
                return False
            if restraint.coordination_maximum is not None and value > restraint.coordination_maximum:
                return False
            return True

        matching = sum(matches(value) for value in coordination.values())
        if restraint.matching_centers == "all":
            if matching != len(coordination):
                return False
        else:
            minimum, maximum = restraint.matching_centers
            if minimum is not None and matching < minimum:
                return False
            if maximum is not None and matching > maximum:
                return False

    return True
