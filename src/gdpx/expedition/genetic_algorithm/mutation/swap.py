#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator

from gdpx.geometry.swap import swap_particles_by_step
from gdpx.utils.atoms_tags import get_tags_per_species


class SwapMutation(OffspringCreator):

    def __init__(
        self,
        bond_distance_dict,
        swap_ratio=0.33,
        covalent_ratio=[0.8, 2.0],
        num_muts=1,
        use_tags=True,
        rng=np.random.default_rng(),
    ):
        """"""
        super().__init__(num_muts=num_muts)
        self.descriptor = "SwMut"
        self.min_inputs = 1

        self.bond_distance_dict = bond_distance_dict
        self.covalent_ratio = covalent_ratio

        self.swap_ratio = swap_ratio

        self.use_tags = use_tags
        if not self.use_tags:
            raise Exception("SwapMutation supports only structure generator with tags.")

        self.rng = rng

        return

    def get_new_individual(self, parents: List[Atoms]):
        """"""
        f = parents[0]

        indi, extra_info = self.mutate(f)
        if indi is None:
            return indi, "mutation: swap"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info["confid"]]

        # finalize_individual, add sub operation descriptor
        indi.info["key_value_pairs"]["origin"] = (
            self.descriptor + "_" + extra_info.split()[0]
        )

        return indi, f"mutation: swap {extra_info}"

    def mutate(self, atoms: Atoms):
        """"""
        mutant = copy.deepcopy(atoms)

        identities = get_tags_per_species(mutant)
        valid_identities = {}
        for k, v in identities.items():
            found_substrate = any([x[0] == 0 for x in v])
            if found_substrate:
                assert len(v) == 1, "We must have only one substrate."
            else:
                valid_identities[k] = v

        num_particle_types = len(valid_identities.keys())

        if num_particle_types > 1:
            num_particle_avg = np.average([len(v) for v in valid_identities.values()])
            num_swaps = int(num_particle_avg*self.swap_ratio)
            mutant, extra_info = swap_particles_by_step(
                mutant,
                identities=valid_identities,
                num_swaps=num_swaps,
                bond_distance_dict=self.bond_distance_dict,
                covalent_ratio=self.covalent_ratio,
                rng=self.rng,
            )
        else:
            mutant, extra_info = None, "NoTwoPType"

        return mutant, extra_info


if __name__ == "__main__":
    ...
