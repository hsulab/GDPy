#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import functools

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.geometry.bounce import bounce_one_atom
from gdpx.group import evaluate_group_expression
from gdpx.utils.atoms_tags import get_tags_per_species


class BounceMutation(OffspringCreator):

    def __init__(
        self,
        particles,
        bond_distance_dict,
        neighlist=None,
        direction="",
        max_disp=2.0,
        covalent_ratio=[0.8, 2.0],
        apply_on_buffer=False,
        num_muts=1,
        use_tags=True,
        rng=np.random.default_rng(),
    ):
        """"""
        super().__init__(num_muts=num_muts)
        self.descriptor = "BnMutation"
        self.min_inputs = 1

        # The particles that can bounce
        self.particles = particles

        self.apply_on_buffer = apply_on_buffer

        # Bounce parameters
        self.direction = direction
        self.max_disp = max_disp

        # Build a neighbour list
        if neighlist is None:
            neighlist = dict(skin=0.0, self_interaction=False, bothways=True)
        self.nlist_prototype = functools.partial(NeighborList, **neighlist)

        self.bond_distance_dict = bond_distance_dict
        self.covalent_ratio = covalent_ratio

        self.use_tags = use_tags
        if not self.use_tags:
            raise Exception("BounceMutation supports only structure generator with tags.")

        self.rng = rng

        return

    def get_new_individual(self, parents: list[Atoms]):
        """"""
        f = parents[0]

        indi, extra_info = self.mutate(f)
        if indi is None:
            return indi, "mutation: bounce"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info["confid"]]

        # finalize_individual, add sub operation descriptor
        indi.info["key_value_pairs"]["origin"] = self.descriptor + "_" + extra_info.split()[0]

        return indi, f"mutation: bounce {extra_info}"

    def mutate(self, atoms: Atoms):
        """Mutate the given atoms.

        The bounce operation will not give a None unless there is no target particle found in the system.

        """
        mutant = copy.deepcopy(atoms)

        # Find a particle to bounce
        if not self.apply_on_buffer:
            # Find particles in the search region that particles have non-zero tags
            identities = get_tags_per_species(mutant)
            valid_identities = {}
            for k, v in identities.items():
                found_substrate = any([x[0] == 0 for x in v])
                if found_substrate:
                    assert len(v) == 1, "We must have only one substrate."
                else:
                    valid_identities[k] = v

            num_particle_types = len(valid_identities.keys())

            if num_particle_types > 0:
                # Get one particle to bounce
                # TODO: Support only atoms for now, thus, atomic_indices for each tag are extended
                valid_atomic_indices = []
                for identity_list in valid_identities.values():
                    for v in identity_list:
                        valid_atomic_indices.extend(v[1])

                num_valid_atomic_indices = len(valid_atomic_indices)
                if num_valid_atomic_indices > 0:
                    particle_index = self.rng.choice(valid_atomic_indices)
                else:
                    particle_index = None
            else:
                num_valid_atomic_indices = 0
                particle_index = None
        else:
            # Find particles in the buffer region (the substrate) by a group expression
            valid_atomic_indices = evaluate_group_expression(mutant, self.particles)
            num_valid_atomic_indices = len(valid_atomic_indices)
            if num_valid_atomic_indices > 0:
                particle_index = self.rng.choice(valid_atomic_indices)
            else:
                num_valid_atomic_indices = 0
                particle_index = None

        if particle_index is not None:
            # Instantiate the neighbour list
            cov_max = self.covalent_ratio[1]
            nlist = self.nlist_prototype(cov_max * np.array(natural_cutoffs(mutant)))

            # bounce the selected particle
            mutant = bounce_one_atom(
                mutant,
                particle_index,
                biased_direction=self.direction,
                max_disp=self.max_disp,
                nlist=nlist,
                covalent_ratio=self.covalent_ratio,
                bond_distance_dict=self.bond_distance_dict,
                rng=self.rng,
                print_func=print,
            )
            extra_info = f"idx_{particle_index}_from_{num_valid_atomic_indices}"
        else:
            mutant = None
            extra_info = "NoPType"

        return mutant, extra_info


if __name__ == "__main__":
    ...
