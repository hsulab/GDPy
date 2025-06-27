#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.data import covalent_radii
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import NeighborList

from gdpx.geometry.composition import convert_string_to_atoms
from gdpx.geometry.exchange import insert_one_particle
from gdpx.geometry.spatial import check_atomic_distances_by_neighbour_list

from .operator import BaseMCOperator


class BasicExchangeOperator(BaseMCOperator):

    MIN_RANDOM_TAG: int = 10000

    MAX_RANDOM_TAG: int = 100000

    def _insert(
        self,
        atoms: Atoms,
        particle: str,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """"""
        # We need covalent bond distanes for neighbour check
        assert hasattr(self, "bond_distance_dict")

        # We cannot use deepcopy here as ase does not delete some arrays,
        # for example, the forces.
        new_atoms = atoms.copy()

        # Prepare particle to add
        adpart = convert_string_to_atoms(particle)

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
        chemicl_numbers = np.hstack([new_atoms.get_atomic_numbers(), adpart.get_atomic_numbers()])
        nlist = self.nlist_prototype(  # type: ignore
            self.covalent_max * np.array([covalent_radii[c] for c in chemicl_numbers])
        )
        check_distance_func = functools.partial(
            check_atomic_distances_by_neighbour_list,
            neighlist=nlist,
        )

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


class ExchangeOperator(BasicExchangeOperator):

    name: str = "exchange"

    #: The current suboperation (insert or remove).
    _curr_operation: Optional[str] = None  # insert or remove

    #: The current tags dict.
    _curr_tags_dict: Optional[dict] = None

    #: The current accpetable volume.
    _curr_volume: Optional[float] = None

    def __init__(
        self,
        particles: list[str],
        chempots: list[float],
        use_bias: bool = True,
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

        self.use_bias = use_bias

        self.nlist_prototype = functools.partial(NeighborList, skin=0.0, self_interaction=False, bothways=True)

        return

    def run(self, atoms: Atoms, rng: np.random.Generator = np.random.default_rng()) -> Optional[Atoms]:
        """"""
        # Check particles in the region
        super().run(atoms)
        self._extra_info = "-"

        # Compute acceptable volume for biased exchange
        if self.use_bias:
            # Determine the exchange volume on-the-fly
            acc_volume = self.region.get_empty_volume(atoms)
        else:
            # Get the volume of the normal region
            acc_volume = self.region.get_volume()
        self._curr_volume = acc_volume

        # Choose a particle to exchange
        particle = self.particles[0]

        assert isinstance(self._curr_tags_dict, dict)
        num_particles = len(self._curr_tags_dict.get(particle, []))

        # Choose insert or remove
        self._print(self.indent + "--> mcattempt")
        if num_particles > 0:
            rn_ex = rng.uniform()
            if rn_ex < 0.5:
                self._print(self.indent + "...insert...")
                self._curr_operation = "insert"
                new_atoms = self._insert(atoms, particle, rng)
            else:
                self._print(self.indent + "...remove...")
                self._curr_operation = "remove"
                new_atoms = self._remove(atoms, particle, rng)
        else:
            self._print(self.indent + "...insert...")
            self._curr_operation = "insert"
            new_atoms = self._insert(atoms, particle, rng)

        return new_atoms

    def metropolis(self, prev_ene: float, curr_ene: float, rng: np.random.Generator = np.random.default_rng()) -> bool:
        """"""
        # Temperature parameters
        kBT_eV = units.kB * self.temperature
        beta = 1.0 / kBT_eV  # 1/(kb*T), eV

        # Compute the cubic thermo de broglie
        particle = self.particles[0]
        chempot = self.chempots[0]

        hplanck = units._hplanck  # J/Hz = kg*m2*s-1
        _species = convert_string_to_atoms(particle)
        _species_mass = np.sum(_species.get_masses())
        _mass = _species_mass * units._amu
        kbT_J = kBT_eV * units._e  # J = kg*m2*s-2
        cubic_wavelength = (hplanck / np.sqrt(2 * np.pi * _mass * kbT_J) * 1e10) ** 3  # thermal de broglie wavelength

        # Compute the prefactor
        # Determine number of exchangeable particles
        assert isinstance(self._curr_tags_dict, dict)
        if particle not in self._curr_tags_dict:
            self._curr_tags_dict[particle] = []
        nexatoms = len(self._curr_tags_dict[particle])

        region_volume = self._curr_volume

        ene_diff = curr_ene - prev_ene
        if self._curr_operation == "insert":
            assert isinstance(region_volume, float)
            prefactor = region_volume / (nexatoms + 1) / cubic_wavelength
            ene_gcmc = ene_diff - chempot
        elif self._curr_operation == "remove":
            prefactor = nexatoms * cubic_wavelength / region_volume
            ene_gcmc = ene_diff + chempot
        else:
            raise RuntimeError(f"Unknown exchange operation {self._curr_operation}.")

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
        self._curr_operation = None
        self._curr_tags_dict = None
        self._curr_volume = None

        return success

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["particles"] = self.particles
        params["chempots"] = self.chempots
        params["use_bias"] = self.use_bias

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
