#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator

from gdpx.geometry.composition import convert_string_to_atoms
from gdpx.geometry.exchange import insert_one_particle, remove_one_particle
from gdpx.nodes.region import RegionVariable
from gdpx.utils.atoms_tags import get_tags_per_species

"""Some mutations that exchange particles with external reservoirs."""


class ExchangeMutation(OffspringCreator):

    #: Maximum number of attempts to rattle atoms.
    MAX_ATTEMPTS: int = 1000

    def __init__(
        self,
        species,
        region,
        bond_distance_dict,
        covalent_ratio=[0.8, 2.0],
        anchors=None,
        nsel=1,
        num_muts=1,
        use_tags=True,
        rng=np.random.default_rng(),
    ):
        """"""
        super().__init__(num_muts=num_muts)
        self.descriptor = "ExMut"
        self.min_inputs = 1

        self.region = RegionVariable(**region).value

        self.bond_distance_dict = bond_distance_dict

        self.covalent_ratio = covalent_ratio

        self.rng = rng

        self.anchors = anchors

        self.nsel = nsel

        self.use_tags = use_tags

        if isinstance(species, str):
            self.species = [species]
        else:  # assume it is a list of chemical formulae
            self.species = species

        self._species_instances = {s: convert_string_to_atoms(s) for s in self.species}

        return

    def get_new_individual(self, parents: List[Atoms]):
        """"""
        f = parents[0]

        indi, extra_info = self.mutate(f)
        if indi is None:
            return indi, "mutation: exchange"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info["confid"]]

        # finalize_individual, add sub operation descriptor
        indi.info["key_value_pairs"]["origin"] = (
            self.descriptor + "_" + extra_info.split()[0]
        )

        return indi, f"mutation: exchange {extra_info}"

    def mutate(self, atoms: Atoms):
        """"""
        mutant = copy.deepcopy(atoms)

        identities = get_tags_per_species(mutant)
        valid_identities = {}
        for k in self.species:
            v = identities.get(k, {})
            if len(v) > 0:
                valid_identities[k] = v

        species_to_exchange = str(self.rng.choice(self.species, replace=False))
        if len(valid_identities) == 0:
            op = "insert"
        else:
            op = self.rng.choice(["insert", "remove"], 1, replace=False)[0]
            if op == "remove":
                species_to_exchange = str(
                    self.rng.choice(list(valid_identities.keys()), replace=False)
                )
            else:
                ...

        extra_info = ""
        if op == "insert":
            mutant, extra_info = insert_one_particle(
                mutant,
                self._species_instances[species_to_exchange],
                region=self.region,
                covalent_ratio=self.covalent_ratio,
                bond_distance_dict=self.bond_distance_dict,
                max_attempts=self.MAX_ATTEMPTS,
                rng=self.rng,
            )
        elif op == "remove":
            mutant, extra_info = remove_one_particle(
                mutant, valid_identities, species_to_exchange, rng=self.rng
            )
        else:
            ...  # Should not be here.

        return mutant, extra_info
    

if __name__ == "__main__":
    ...
