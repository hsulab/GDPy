import functools
from typing import Any, Callable, Literal, Optional

from ase import Atoms

from gdpx.structures.groups.group import evaluate_group_expression
from gdpx.structures.regions.factory import create_region
from gdpx.structures.geometry.restraints import evaluate_restraints, parse_restraints
from gdpx.utils.atoms_tags import get_tags_per_species


def extinct_by_region(
    atoms: Atoms, group: str, region,
) -> Literal[0, 1]:
    """Extinguish when any selected atom lies in the forbidden region.

    Selection uses the original structure. The region owns membership,
    coordinate wrapping, and boundary conventions.
    """
    indices = set(evaluate_group_expression(atoms, group))
    if not indices:
        return 0
    candidate = atoms.copy()
    region.preprocess(candidate)
    return int(bool(indices.intersection(region.get_contained_indices(candidate))))


def extinct_by_restraints(atoms: Atoms, restraints) -> Literal[0, 1]:
    """Mark a structure extinct when any geometric restraint fails."""
    return int(not evaluate_restraints(atoms, restraints))


def extinct_by_number_of_particles(
    atoms: Atoms,
    particle: str,
    min_num: Optional[int] = None,
    max_num: Optional[int] = None,
) -> Literal[0, 1]:
    """"""
    identities = get_tags_per_species(atoms)
    num_particles = len(identities.get(particle, []))

    extinct = 0
    if min_num is not None and num_particles < min_num:
        extinct = 1
    if max_num is not None and num_particles > max_num:
        extinct = 1

    return extinct


def dispatch_thanos(
    name: str,
    restraints: Optional[list[dict[str, Any]]] = None,
    covalent_ratio=(0.8, 2.0),
    number: tuple[str, Optional[int], Optional[int]] = ("", None, None),
    group: Optional[str] = None,
    region: Optional[dict[str, Any]] = None,
) -> Callable[[Atoms], Literal[0, 1]]:
    """"""
    if name == "region":
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Region Thanos requires a non-empty group expression.")
        if region is None:
            raise ValueError("Region Thanos requires a region configuration.")
        return functools.partial(extinct_by_region, group=group, region=create_region(region))
    elif name == "restraints":
        parsed_restraints = parse_restraints(restraints, covalent_ratio=covalent_ratio)
        return functools.partial(extinct_by_restraints, restraints=parsed_restraints)
    elif name == "particle_number":
        particle, min_num, max_num = number
        return functools.partial(extinct_by_number_of_particles, particle=particle, min_num=min_num, max_num=max_num)
    else:
        raise Exception(f"Thanos function '{name}' is not recognised.")
