"""Atomic-type and bond-distance preparation shared by move consumers."""

from typing import Optional

from ase import Atoms
from ase.data import atomic_numbers
from ase.formula import Formula

from gdpx.structures.geometry.spatial import get_bond_distance_dict


def prepare_operators(operators, atomic_numbers, bond_distances=None, custom_pairs=None):
    """Attach geometry data without taking ownership of an Atoms object."""
    distances = dict(bond_distances or {})
    distances.update(get_bond_distance_dict(atomic_numbers, ratio=1.0))
    for operator in operators:
        operator.bond_distance_dict = distances
        operator.blmin = {pair: value * operator.covalent_min for pair, value in distances.items()}
        operator.custom_pair_distance_dict = custom_pairs


def infer_unique_atomic_numbers(
    operators,
    custom_atomic_types: Optional[list[str]] = None,
    substrates: Optional[list[Atoms]] = None,
) -> list[int]:
    """Find possible elements in the simulation and build a bond-distance list."""
    type_list = []
    for op in operators:
        # TODO: wee need further unify the names here
        if hasattr(op, "particles"):
            for p in op.particles:
                type_list.extend(list(Formula(p).count().keys()))
        elif hasattr(op, "species"):
            type_list.extend(list(Formula(op.species).count().keys()))
        elif hasattr(op, "reservoir"):
            type_list.extend(list(Formula(op.reservoir["species"]).count().keys()))
        elif hasattr(op, "reaction"):
            for species in op.reaction.particles:
                type_list.extend(Formula(species).count())
        else:
            ...
    if custom_atomic_types is not None:
        type_list.extend(custom_atomic_types)
        type_list = list(set(type_list))
    if substrates is not None:
        for atoms in substrates:
            type_list = list(set(type_list + atoms.get_chemical_symbols()))
    unique_atomic_numbers = sorted({atomic_numbers[a] for a in type_list})

    return unique_atomic_numbers
