#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import functools
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.neighborlist import NeighborList

from gdpx.geometry.composition import convert_string_to_atoms

from ..operator import BaseMCOperator
from ..statmech import compute_thermo_wavelength


class BasicExchangeOperator(BaseMCOperator):

    name: str = "exchange"

    #: The current tags dict.
    _curr_tags_dict: Optional[dict] = None

    def __init__(
        self,
        particles: list[str],
        chempots: list[float],
        skip_distance_check: bool = False,
        *args,
        **kwargs,
    ):
        """"""
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

        self.particles = particles
        self.chempots = chempots

        # Get exchangeable particle atoms based on its name
        _particle_instances = []
        for particle in self.particles:
            _particle_instance = convert_string_to_atoms(particle)
            if _particle_instance is None:
                raise Exception(f"Particle {particle} is not a valid chemical symbol or a molecule formula.")
            _particle_instances.append(_particle_instance)
        self._particle_instances = _particle_instances

        _cubic_wavelengths = []
        for _particle_instance in self._particle_instances:
            _cubic_wavelength = compute_thermo_wavelength(
                mass=_particle_instance.get_masses().sum(),
                temperature=self.temperature,
            )
            _cubic_wavelengths.append(_cubic_wavelength)
        self._cubic_wavelengths = _cubic_wavelengths

        # Check if neighbour distances should be checked
        self.skip_distance_check = skip_distance_check

        self.nlist_prototype = functools.partial(NeighborList, skin=0.0, self_interaction=False, bothways=True)

        # Some state information after mc attempts and before energy evaluation
        self._atoms = None
        self._state = {}

        return

    def run(self, atoms: Atoms, rng: np.random.Generator = np.random.default_rng()) -> Optional[Atoms]:
        """"""
        # Check particles in the region
        super().run(atoms)
        self._extra_info = "-"

        self._state["volume"] = self._update_volume(atoms)

        # Choose a particle to exchange
        particle = self.particles[0]
        particle_instance = self._particle_instances[0]

        assert isinstance(self._curr_tags_dict, dict)
        num_particles = len(self._curr_tags_dict.get(particle, []))

        # Choose insert or remove
        self._print(self.indent + "--> mcattempt")
        self._print(self.indent + f"check distance: {not self.skip_distance_check}")
        if num_particles > 0:
            rn_ex = rng.uniform()
            if rn_ex < 0.5:
                self._print(self.indent + "...insert...")
                self._state["operation"] = "insert"
                new_atoms = self._insert(atoms, particle, particle_instance, rng)
            else:
                self._print(self.indent + "...remove...")
                self._state["operation"] = "remove"
                new_atoms = self._remove(atoms, particle, rng)
        else:
            self._print(self.indent + "...insert...")
            self._state["operation"] = "insert"
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

    def metropolis(self, prev_ene: float, curr_ene: float, rng: np.random.Generator = np.random.default_rng()) -> bool:
        """"""
        # Temperature parameters
        kBT_eV = units.kB * self.temperature
        beta = 1.0 / kBT_eV  # 1/(kb*T), eV

        # Get particle properties
        particle = self.particles[0]
        chempot = self.chempots[0]
        cubic_wavelength = self._cubic_wavelengths[0]

        # Compute the prefactor
        # Determine number of exchangeable particles
        assert isinstance(self._curr_tags_dict, dict)
        if particle not in self._curr_tags_dict:
            self._curr_tags_dict[particle] = []
        nexatoms = len(self._curr_tags_dict[particle])

        region_volume = self._state["volume"]

        ene_diff = curr_ene - prev_ene
        if self._state["operation"] == "insert":
            assert isinstance(region_volume, float)
            prefactor = region_volume / (nexatoms + 1) / cubic_wavelength
            ene_gcmc = ene_diff - chempot
        elif self._state["operation"] == "remove":
            prefactor = nexatoms * cubic_wavelength / region_volume
            ene_gcmc = ene_diff + chempot
        else:
            raise RuntimeError(f"Unknown exchange operation {self._state['operation']}.")

        acc_ratio = np.min([1.0, prefactor * np.exp(-beta * (ene_gcmc))])
        ran_ratio = rng.uniform()

        # Some log information
        content = "--> mcstate\n"
        content += f"Volume {region_volume:>12.4f} [A^3] Beta {beta:>12.4f} [1/eV]\n"
        content += f"Prefactor {prefactor:>12.4f}\n"
        content += f"CubicWavelength {cubic_wavelength:>12.4f} [A^3]\n"
        content += f"Nexatoms {nexatoms:>12d}\n"
        content += f"dE {ene_diff:>12.4f} [eV]  " + f"dF {ene_gcmc:>12.4f} [eV]\n"
        content += f"Accept {acc_ratio:>4.2e} >? {ran_ratio:>4.2e}"
        for x in content.split("\n"):
            self._print(self.indent + x)

        success = ran_ratio < acc_ratio

        # Clear state
        self._state = {}
        self._atoms = None

        return success

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["particles"] = self.particles
        params["chempots"] = self.chempots
        params["skip_distance_check"] = self.skip_distance_check

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

        return content


if __name__ == "__main__":
    ...
