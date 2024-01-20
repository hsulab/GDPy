#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
from typing import NoReturn, Callable, List

import numpy as np

from ase import Atoms

from gdpx.core.register import registers


class AbstractOperator(abc.ABC):

    #: Operator name.
    name: str = "abstract"

    #: Maximum attempts to generate the new position of a species.
    MAX_RANDOM_ATTEMPTS = 1000

    #: Print function.
    _print: Callable = print

    def __init__(
            self, region: dict={}, temperature: float=300., pressure: float=1., 
            covalent_ratio=[0.8, 2.0], use_rotation=True, prob: str=1.0, *args, **kwargs
        ) -> None:
        """Initialise the modification operator.

        Args:
            temperature: Monte Carlo temperature [K].
            pressure: Monte Carlo pressure [bar].
            covalent_ratio: Minimum and maximum percentages of covalent distance.

        """
        super().__init__()

        # - region
        region_params = copy.deepcopy(region)
        region_method = region_params.pop("method", "auto")
        self.region = registers.create("region", region_method, convert_name=True, **region_params)

        # - thermostat
        self.temperature = temperature
        self.pressure = pressure

        # - restraint on atomic distance
        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        # - molecule
        self.use_rotation = use_rotation

        # - probability
        self.prob = prob

        return
    
    def _check_region(self, atoms: Atoms, *args, **kwargs):
        """"""
        if self.region.__class__.__name__ == "AutoRegion":
            self.region._curr_atoms = atoms

        # NOTE: Modify only atoms in the region...
        tags_dict = self.region.get_tags_dict(atoms)
        content = "species within system:\n"
        content += "  " + "  ".join([str(k)+" "+str(len(v)) for k, v in tags_dict.items()]) + "\n"
        for x in content.split("\n"):
            self._print(x)

        self._curr_tags_dict = self.region.get_contained_tags_dict(atoms, tags_dict)
        content = "species within region:\n"
        content += "  " + "  ".join([str(k)+" "+str(len(v)) for k, v in self._curr_tags_dict.items()]) + "\n"
        for x in content.split("\n"):
            self._print(x)

        return
    
    def _select_species(self, atoms: Atoms, particles: List[str]=None, rng=np.random) -> List[int]:
        """"""
        # - pick a particle (atom/molecule)
        tags_within_region = []
        for k, v in self._curr_tags_dict.items():
            if particles is not None and k not in particles:
                continue
            tags_within_region.extend(v)
        self._print(f"ntags of {particles}: {len(tags_within_region)}")

        if len(tags_within_region) > 0:
            picked_tag = rng.choice(tags_within_region)
            tags = atoms.get_tags()
            species_indices = [i for i, t in enumerate(tags) if t==picked_tag]
            self._print(f"selected tag: {picked_tag} species: {atoms[species_indices].get_chemical_formula()}")
        else:
            picked_tag = None
            species_indices = None
            raise RuntimeError(f"{self.__class__.__name__} does not have {particles}.")

        return species_indices

    def _rotate_species(self, species: Atoms, rng=np.random):
        """"""
        species_ = species.copy() # TODO: make clean atoms?
        org_com = np.mean(species_.positions, axis=0)
        if self.use_rotation and len(species_) > 1:
            phi, theta, psi = 360 * rng.uniform(0,1,3)
            species_.euler_rotate(
                phi=phi, theta=0.5 * theta, psi=psi,
                center=org_com
            )

        return species_
    
    @abc.abstractmethod
    def run(self, atoms: Atoms, rng=np.random) -> Atoms:
        """Modify the input atoms.

        Returns:
            A new atoms is returned if the modification succeeds otherwise None is
            returned.

        """
        self._check_region(atoms)

        return
    
    @abc.abstractmethod
    def metropolis(self, prev_ene: float, curr_ene: float, rng=np.random) -> bool:
        """Monte Carlo."""

        return
    
    def as_dict(self) -> dict:
        """"""
        params = {}
        params["method"] = self.name
        params["region"] = self.region.as_dict()
        params["temperature"] = self.temperature
        params["pressure"] = self.pressure
        params["covalent_ratio"] = [self.covalent_min, self.covalent_max]
        params["use_rotation"] = self.use_rotation
        params["prob"] = self.prob

        params = copy.deepcopy(params)

        return params


if __name__ == "__main__":
    ...