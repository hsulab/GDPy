#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List

import numpy as np

from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator

from .. import registers
from ..utils import convert_string_to_atoms
from .. import get_tags_per_species
from ..utils import check_overlap_neighbour

"""Some mutations that exchange particles with external reservoirs."""


def rotate_molecule(molecule: Atoms, rng=np.random.default_rng()) -> Atoms:
    """Rotate a molecule randomly."""
    org_com = molecule.get_center_of_mass()
    if len(molecule) > 1:
        phi, theta, psi = 360 * rng.uniform(0, 1, 3)
        molecule.euler_rotate(phi=phi, theta=0.5 * theta, psi=psi, center=org_com)
    else:
        ...

    return molecule


class ExchangeMutation(OffspringCreator):

    #: Maximum number of attempts to rattle atoms.
    MAX_ATTEMPTS: int = 1000

    def __init__(
        self,
        species,
        region,
        anchors=None,
        nsel=1,
        covalent_ratio=[0.8, 2.0],
        num_muts=1,
        rng=np.random.default_rng(),
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(num_muts=num_muts, rng=rng, *args, **kwargs)
        self.descriptor = "ExchangeMutation"
        self.min_inputs = 1

        self.species = species

        region_params = copy.deepcopy(region)
        region_method = region_params.pop("method", "auto")
        self.region = registers.create(
            "region", region_method, convert_name=True, **region_params
        )

        self.anchors = anchors

        self.nsel = nsel

        self.covalent_ratio = covalent_ratio

        self._species_instances = []
        for s in self.species:
            self._species_instances.append(convert_string_to_atoms(s))

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
        # print(f"{valid_identities =}")

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
            mutant, extra_info = self._insert(
                mutant, valid_identities, species_to_exchange
            )
        elif op == "remove":
            mutant, extra_info = self._remove(
                mutant, valid_identities, species_to_exchange
            )
        else:
            ...  # Should not be here.

        return mutant, extra_info

    def _insert(self, atoms: Atoms, identities: dict, species: str):
        """"""
        clean_adsorbate: Atoms = copy.deepcopy(
            self._species_instances[self.species.index(species)]
        )
        ads_tag = int(np.max(atoms.get_tags()) + 17)
        clean_adsorbate.set_tags(ads_tag)
        num_atoms_adsorbate = len(clean_adsorbate)
        mutant = copy.deepcopy(atoms) + clean_adsorbate

        if self.anchors is None:
            # place the adsorbate in a random location
            for _ in range(self.MAX_ATTEMPTS):
                adsorbate = copy.deepcopy(clean_adsorbate)
                adsorbate = rotate_molecule(adsorbate, rng=self.rng)
                old_com = adsorbate.get_center_of_mass()
                ran_pos = self.region.get_random_positions(size=1, rng=self.rng)[0]
                adsorbate.translate(ran_pos - old_com)
                mutant.positions[-num_atoms_adsorbate:] = adsorbate.positions
                if check_overlap_neighbour(mutant, self.covalent_ratio):
                    break
            else:
                mutant = None  # Failed to insert.
        else:
            # TODO: add the adsorbate on one of the anchors
            ...

        return mutant, f"insert {species}"

    def _remove(self, atoms: Atoms, identities: dict, species: str):
        """"""
        selected = self.rng.choice(range(len(identities[species])), replace=False)
        selected_indices = identities[species][selected][1]
        del atoms[selected_indices]

        return atoms, f"remove {species} {selected_indices}"


if __name__ == "__main__":
    ...
