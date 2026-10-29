"""BaseRegion implementation."""

import abc
from typing import Mapping, Optional

import numpy as np
from ase import Atoms, data

from gdpx.utils.atoms_tags import get_tags_per_species


class BaseRegion(abc.ABC):
    """The base class of region."""

    def __init__(self, origin: list[float]):
        """"""
        self._origin = np.array(origin)

        return

    @staticmethod
    def from_str(command: str) -> "BaseRegion":
        """Init a region from the command"""

        ...

    def preprocess(self, atoms: Atoms):
        """Preprocess the region with the atoms."""
        if hasattr(self, "_atoms"):
            self._atoms = atoms

        return

    def get_contained_indices(self, atoms: Atoms):
        """"""
        indices_within_region = []
        for i, a in enumerate(atoms):
            if self._is_within_region(a.position):
                indices_within_region.append(i)

        return indices_within_region

    def get_random_positions(self, size=1, rng=np.random.default_rng()):
        """"""
        random_positions = []
        for _ in range(size):
            ran_pos = self._get_a_random_position(rng)
            random_positions.append(ran_pos)
        random_positions = np.array(random_positions)

        return random_positions

    @abc.abstractmethod
    def _get_a_random_position(self, rng: np.random.Generator) -> np.ndarray:
        """"""

        ...

    @abc.abstractmethod
    def _is_within_region(self, position) -> bool:
        """Positions are normally atomic positions or molecular centre positions."""

        ...

    def get_tags_dict(self, atoms: Atoms):
        """Get tags dict for atoms within the (entire) system"""

        return get_tags_per_species(atoms)

    def _get_positions_for_tags(self, atoms: Atoms):
        """Coordinates used to locate tagged particle centers."""
        return atoms.get_positions(wrap=True)

    def get_contained_tags_dict(self, atoms: Atoms, tags_dict: Optional[dict] = None) -> Mapping[str, list[int]]:
        """"""
        # - find tags and compute cops
        if tags_dict is None:
            tags_dict_within_system = get_tags_per_species(atoms)
        else:
            tags_dict_within_system = tags_dict

        # Get wrapped positions due to PBC
        positions = self._get_positions_for_tags(atoms)

        cops_dict = {}
        for key, tags_and_indices in tags_dict_within_system.items():
            for tag, curr_indices in tags_and_indices:
                cur_positions = positions[curr_indices]
                # TODO: Considering PBC, surface may have molecules across boundaries.
                cop = np.average(cur_positions, axis=0)
                if key not in cops_dict:
                    cops_dict[key] = []
                cops_dict[key].append([tag, cop])

        # Check if the fragment is within the region
        tags_dict_within_region = {}
        for key, tags_and_cops in cops_dict.items():
            # print(tags_and_cops)
            for tag, cop in tags_and_cops:
                if self._is_within_region(cop):
                    if key in tags_dict_within_region:
                        tags_dict_within_region[key].append(tag)
                    else:
                        tags_dict_within_region[key] = [tag]

        return tags_dict_within_region

    def get_empty_volume(
        self,
        atoms: Atoms,
        tags_dict: Optional[dict] = None,
        ratio: float = 1.0,
    ) -> float:
        """Empty volume = Region volume - total volume of atoms within region.

        This is not always correct since all atoms in the fragment are considered
        within the region if their cop is in the region.

        """
        # Get atom indices with given tags
        if tags_dict is None:
            tags_dict = self.get_contained_tags_dict(atoms)
        tags_within_region = []
        for key, tags in tags_dict.items():
            tags_within_region.extend(tags)
        atomic_indices = [i for i, t in enumerate(atoms.get_tags()) if t in tags_within_region]

        # Get atoms' radii
        radii = np.array([data.covalent_radii[data.atomic_numbers[atoms[i].symbol]] for i in atomic_indices])
        radii *= ratio

        atoms_volume = np.sum([4.0 / 3.0 * np.pi * r**3 for r in radii])

        return self.get_volume() - atoms_volume

    @abc.abstractmethod
    def get_volume(self) -> float:
        """"""

        ...

    @abc.abstractmethod
    def as_dict(self) -> dict:
        """"""

        ...
