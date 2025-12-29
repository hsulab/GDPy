from typing import Optional

import numpy as np
from ase import Atoms

from gdpx.group import evaluate_group_expression

from .comparator import BaseComparator


def point_mass_inertia_tensor(mass: float, position: np.ndarray) -> np.ndarray:
    """Function to calculate the inertia tensor for a point mass."""
    I = np.zeros((3, 3))
    r_squared = np.dot(position, position)

    for i in range(3):
        for j in range(3):
            if i == j:
                I[i, j] = mass * (r_squared - position[i] ** 2)
            else:
                I[i, j] = -mass * position[i] * position[j]

    return I


def calculate_inertia_tensor(coordinates: np.ndarray, atomic_masses: np.ndarray) -> np.ndarray:
    """Function to calculate the total inertia tensor for the nanoparticle."""
    total_inertia_tensor = np.zeros((3, 3))

    # Iterate through each copper atom and add its contribution to the total inertia tensor
    for i in range(len(coordinates)):
        atom_position, atomic_mass = coordinates[i], atomic_masses[i]
        atom_inertia_tensor = point_mass_inertia_tensor(atomic_mass, atom_position)
        total_inertia_tensor += atom_inertia_tensor

    return total_inertia_tensor


class InertiaTensorComparator(BaseComparator):
    def __init__(self, group: Optional[str], *args, **kwargs):
        """Initialise the comparator."""
        super().__init__(*args, **kwargs)

        self.group = group

        return

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """"""
        is_similar = super().looks_like(a1, a2)
        if is_similar:
            group_indices = list(range(len(a1)))  # number of atoms have been checked to be the same
            if self.group is not None:
                g1 = evaluate_group_expression(a1, self.group)
                g2 = evaluate_group_expression(a2, self.group)
                if g1 == g2:  # can be []
                    group_indices = g1
                else:
                    group_indices = []
                num_group_atoms = len(group_indices)
                if num_group_atoms > 0:
                    ...
                else:
                    ...
            else:
                ...
        else:
            ...

        return is_similar
