#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import functools
from typing import Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList

from gdpx.structures.geometry.composition import convert_string_to_adsorbate, convert_string_to_atoms

from ..operator import BaseMCOperator


class BasicExchangeOperator(BaseMCOperator):

    name: str = "exchange"

    #: The current tags dict.
    _curr_tags_dict: Optional[dict] = None

    def __init__(
        self,
        particles: list[str],
        chempots: list[float],
        use_ads: bool = False,
        *args,
        **kwargs,
    ):
        """Initialise the basic exchange operator.

        Args:
            particles: List of particle names to exchange.
            chempots: List of chemical potentials for the particles [eV].
            use_ads: Whether to use adsorbate conversion for particles.

        """
        super().__init__(
            *args,
            **kwargs,
        )

        num_particles = len(particles)
        if num_particles != 1:
            raise Exception(f"Exchange operator requires exactly one particle, got {num_particles}.")

        num_chempots = len(chempots)
        if num_chempots != 1:
            raise Exception(f"Exchange operator requires exactly one chemical potential, got {num_chempots}.")

        self.use_ads = use_ads
        self.particles = particles
        self.chempots = chempots

        # Get exchangeable particle atoms based on its name
        convert_func = convert_string_to_adsorbate if use_ads else convert_string_to_atoms

        _particle_instances = []
        for particle in self.particles:
            _particle_instance = convert_func(particle)
            if _particle_instance is None:
                raise Exception(f"Particle {particle} is not a valid chemical symbol or a molecule formula.")
            _particle_instances.append(_particle_instance)
        self._particle_instances = _particle_instances

        # Neighbor list prototype
        self.nlist_prototype = functools.partial(NeighborList, skin=0.0, self_interaction=False, bothways=True)

        return

    def _propose(self, atoms: Atoms, rng: np.random.Generator = np.random.default_rng()) -> Optional[Atoms]:
        """"""
        # Check particles in the region
        super()._propose(atoms, rng)
        self._extra_info = "-"

        # Choose a particle to exchange
        particle = self.particles[0]
        particle_instance = self._particle_instances[0]

        assert isinstance(self._curr_tags_dict, dict)
        num_particles = len(self._curr_tags_dict.get(particle, []))

        self._state["num_particles"] = num_particles
        self._state["volume"] = self._update_volume(atoms)

        # Choose insert or remove
        self._print(self.indent + "--> mcattempt")
        self._print(self.indent + f"check distance: {not self.skip_distance_check}")
        if num_particles > 0:
            rn_ex = rng.uniform()
            if rn_ex < 0.5:
                self._print(self.indent + "...insert...")
                self._state["operation"] = "insert"
                self._state["proposal_ratio"] = 1.0
                new_atoms = self._insert(atoms, particle, particle_instance, rng)
            else:
                self._print(self.indent + "...remove...")
                self._state["operation"] = "remove"
                # The reverse move is forced insertion after removing the last
                # particle, whereas this deletion branch was selected with 1/2.
                self._state["proposal_ratio"] = 2.0 if num_particles == 1 else 1.0
                new_atoms = self._remove(atoms, particle, rng)
        else:
            self._print(self.indent + "...insert...")
            self._state["operation"] = "insert"
            # Insertion is forced here, but the reverse deletion from N=1 is
            # selected with probability 1/2.
            self._state["proposal_ratio"] = 0.5
            new_atoms = self._insert(atoms, particle, particle_instance, rng)

        return new_atoms

    @abc.abstractmethod
    def _update_volume(self, atoms: Atoms) -> float:
        """Update the volume of the region based on the current atoms."""
        ...

    @abc.abstractmethod
    def _insert(
        self, atoms: Atoms, particle: str, particle_instance: Atoms, rng: np.random.Generator = np.random.default_rng()
    ) -> Optional[Atoms]: ...

    @abc.abstractmethod
    def _remove(self, atoms: Atoms, particle: str, rng: np.random.Generator = np.random.default_rng()) -> Atoms: ...


    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["particles"] = self.particles
        params["chempots"] = self.chempots
        params["use_ads"] = self.use_ads

        return params

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"particles: \n"
        content += f"  {self.particles}\n"
        content += f"chempots: \n"
        content += f"  {self.chempots}\n"
        content += f"within the region {self.region}\n"

        # add indent
        content = self.indent + content.replace("\n", "\n" + self.indent)

        return content


if __name__ == "__main__":
    ...
