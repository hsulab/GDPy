#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import functools

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from gdpx.geometry.exchange import insert_one_particle
from gdpx.geometry.spatial import check_atomic_distances_by_neighbour_list

from .basic import BasicExchangeOperator


class ExchangeOperator(BasicExchangeOperator):

    MIN_RANDOM_TAG: int = 10000

    MAX_RANDOM_TAG: int = 100000

    def _update_volume(self, atoms: Atoms) -> float:
        """Update the volume of the region based on the current atoms."""
        # Determine the exchange volume on-the-fly
        acc_volume = self.region.get_volume()

        return acc_volume

    def _insert(
        self,
        atoms: Atoms,
        particle: str,
        particle_instance: Atoms,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """"""
        # We need covalent bond distanes for neighbour check
        assert hasattr(self, "bond_distance_dict")

        # We cannot use deepcopy here as ase does not delete some arrays,
        # for example, the forces.
        self._atoms = atoms
        new_atoms = atoms.copy()

        # Prepare particle to add
        adpart = copy.deepcopy(particle_instance)

        # Add velocity in case the mixed MC/MD is performed
        MaxwellBoltzmannDistribution(adpart, temperature_K=self.temperature, rng=rng)

        # Choose a tag for the particle
        used_tags = set(atoms.get_tags().tolist())
        adpart_tag = 0
        while adpart_tag in used_tags:
            adpart_tag = rng.integers(self.MIN_RANDOM_TAG, self.MAX_RANDOM_TAG)
        adpart_tag = int(adpart_tag)
        self._print(self.indent + f"adpart {adpart.get_chemical_formula()} tag: {adpart_tag} {type(adpart_tag)}")

        # Use neighbour list
        if not self.skip_distance_check:
            chemicl_numbers = np.hstack([new_atoms.get_atomic_numbers(), adpart.get_atomic_numbers()])
            nlist = self.nlist_prototype(  # type: ignore
                self.covalent_max * np.array([covalent_radii[c] for c in chemicl_numbers])
            )
            check_distance_func = functools.partial(
                check_atomic_distances_by_neighbour_list,
                neighlist=nlist,
            )
        else:
            check_distance_func = None

        # Insert the particle
        new_atoms, info = insert_one_particle(
            atoms=new_atoms,
            particle=adpart,
            region=self.region,
            covalent_ratio=[self.covalent_min, self.covalent_max],
            bond_distance_dict=self.bond_distance_dict,  # type: ignore
            particle_tag=adpart_tag,
            sort_tags=False,
            max_attempts=self.MAX_RANDOM_ATTEMPTS,
            check_distance_func=check_distance_func,
            rng=rng,
        )

        _, _, state, num_attempts = info.split("_")
        if state == "success":
            self._print(self.indent + f"succeed to insert after {num_attempts} attempts...")
            self._extra_info = f"Insert_{particle}_{adpart_tag}"  # type: ignore
        elif state == "failure":
            self._print(self.indent + f"failed to insert after {num_attempts} attempts...")
        else:
            raise Exception("This should not happen.")

        return new_atoms

    def _remove(
        self,
        atoms: Atoms,
        particle: str,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> Atoms:
        """"""
        # We cannot use deepcopy here as ase does not delete some arrays,
        # for example, the forces.
        self._atoms = atoms
        new_atoms = atoms.copy()

        # Pick one random particle
        species_indices = self._select_species(new_atoms, [particle], rng)

        # The tags for atoms in the particle should be the same, we need check this?
        particle_tag = new_atoms.get_tags()[species_indices][0]

        # Remove then
        del new_atoms[species_indices]

        # Update info
        self._extra_info = f"Remove_{particle}_{particle_tag}"  # type: ignore

        return new_atoms

    def revert_state(self, atoms: Atoms) -> None:
        """"""
        operation = self._state.get("operation")
        if operation == "insert":
            ...
        elif operation == "remove":
            ...
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return


class BiasedVolumeExchangeOperator(ExchangeOperator):
    """Biased volume exchange operator.

    This operator is used to insert or remove particles in a biased volume.
    The biased volume is defined by the region volume not occupied by existing particles.

    """

    name: str = "biased_volume_exchange"

    def _update_volume(self, atoms: Atoms) -> float:
        """Update the volume of the region based on the current atoms."""
        # Determine the exchange volume on-the-fly
        acc_volume = self.region.get_empty_volume(atoms)

        return acc_volume


if __name__ == "__main__":
    ...
