#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import functools
from typing import Optional, Union

import ase.data
import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from gdpx.geometry.exchange import insert_one_particle_on_site
from gdpx.geometry.spatial import check_atomic_distances_by_neighbour_list
from gdpx.graph.adsorption import find_adsorption_sites_by_graph

from .basic import BasicExchangeOperator


class AdsorbateExchangeOperator(BasicExchangeOperator):
    name: str = "adsorbate_exchange"

    MIN_RANDOM_TAG: int = 10000

    MAX_RANDOM_TAG: int = 100000

    def __init__(
        self,
        particles: list[str],
        chempots: list[float],
        anchors: Union[dict, list[dict]],
        use_ads: bool = True,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(
            particles=particles,
            chempots=chempots,
            use_ads=use_ads,
            *args,
            **kwargs,
        )

        # We must use adsorbate as it contains how to bind to surfaces
        if not use_ads:
            raise Exception("AdsorbateExchangeOperator requires use_ads=True.")

        num_particles = len(self.particles)
        if isinstance(anchors, list):
            num_anchors = len(anchors)
            assert num_anchors == num_particles, f"num_anchors {num_anchors} != num_particles {num_particles}"
            self.anchors = dict()
            for particle, anchor in zip(self.particles, anchors):
                self.anchors[particle] = anchor
        else:
            self.anchors = dict(
                _default=anchors,
            )

        return

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
    ) -> Optional[Atoms]:
        """"""
        # We need covalent bond distanes for neighbour check
        assert hasattr(self, "bond_distance_dict")

        assert hasattr(self, "custom_pair_distance_dict")
        custom_pair_distance_dict = self.custom_pair_distance_dict if self.custom_pair_distance_dict else None  # type: ignore

        # We cannot use deepcopy here as ase does not delete some arrays,
        # for example, the forces.
        self._atoms = atoms
        new_atoms = atoms

        # Prepare particle to add
        adpart = copy.deepcopy(particle_instance)

        atomic_indices = list(range(len(new_atoms), len(new_atoms) + len(adpart)))
        self._state["atomic_indices"] = atomic_indices

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
                self.covalent_max * np.array([ase.data.covalent_radii[c] for c in chemicl_numbers])
            )
            check_distance_func = functools.partial(
                check_atomic_distances_by_neighbour_list,
                neighlist=nlist,
            )
        else:
            check_distance_func = None

        # Create the find site function
        site_params = self.anchors.get(particle, self.anchors["_default"])

        find_sites_func = functools.partial(
            find_adsorption_sites_by_graph,
            group_expr=site_params.get("group"),
            cutoff=site_params.get("cutoff", 3.0),
            max_order=site_params.get("max_order", 3),
            surf_index=site_params.get("surf_index", 2),
        )

        # Insert the particle
        _, info = insert_one_particle_on_site(
            atoms=new_atoms,
            particle=adpart,
            find_sites_func=find_sites_func,
            covalent_ratio=(self.covalent_min, self.covalent_max),
            bond_distance_dict=self.bond_distance_dict,  # type: ignore
            custom_pair_distance_dict=custom_pair_distance_dict,
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
            # If adsorb failed, 
            # no revert is needed as it has been done by the function above.
            # del new_atoms[atomic_indices]
            self._extra_info = f"Insert_Failed"
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
        new_atoms = atoms

        # Pick one random particle
        atomic_indices = self._select_species(new_atoms, [particle], rng)

        removed_particle = new_atoms[atomic_indices]
        assert isinstance(removed_particle, Atoms), "Removed particle should be an Atoms object."
        tags = removed_particle.get_tags()
        assert len(set(tags)) == 1, "All tags for the selected atoms should be the same."
        particle_tag = tags[0]

        self._state["removed_particle"] = removed_particle

        # Remove then
        del new_atoms[atomic_indices]

        # Update info
        self._extra_info = f"Remove_{particle}_{particle_tag}"  # type: ignore

        return new_atoms

    def revert_state(self, atoms: Atoms) -> None:
        """"""
        operation = self._state.get("operation")
        if operation == "insert":
            atomic_indices = self._state.get("atomic_indices")
            del atoms[atomic_indices]
        elif operation == "remove":
            # The removed particle will be added to the end of the atoms,
            # the order of atoms has changed but the tags are preserved.
            removed_particle = self._state.get("removed_particle")
            atoms.extend(removed_particle)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return

    def metropolis(
        self,
        prev_ene: float,
        curr_ene: float,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> bool:
        """This uses a naive metropolis criterion that breaks detailed balance."""
        success = super().metropolis(prev_ene, curr_ene, rng)

        return success


if __name__ == "__main__":
    ...
