#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.data import covalent_radii
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import NeighborList

from gdpx.geometry.exchange import insert_one_particle
from gdpx.geometry.spatial import check_atomic_distances_by_neighbour_list

from .. import convert_string_to_atoms
from .operator import AbstractOperator


class BasicExchangeOperator(AbstractOperator):

    MIN_RANDOM_TAG: int = 10000

    MAX_RANDOM_TAG: int = 100000

    def _insert(
        self,
        atoms: Atoms,
        species: str,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """"""
        # We need covalent bond distanes for neighbour check
        assert hasattr(self, "bond_distance_dict")

        # We cannot use deepcopy here as ase does not delete some arrays,
        # for example, the forces.
        new_atoms = atoms.copy()

        # Prepare particle to add
        adpart = convert_string_to_atoms(species)

        # Add velocity in case the mixed MC/MD is performed
        MaxwellBoltzmannDistribution(adpart, temperature_K=self.temperature, rng=rng)

        # Choose a tag for the particle
        used_tags = set(atoms.get_tags().tolist())
        adpart_tag = 0
        while adpart_tag in used_tags:
            adpart_tag = rng.integers(self.MIN_RANDOM_TAG, self.MAX_RANDOM_TAG)
        adpart_tag = int(adpart_tag)
        self._print(
            f"adpart {adpart.get_chemical_formula()} tag: {adpart_tag} {type(adpart_tag)}"
        )

        # Use neighbour list
        chemicl_numbers = np.hstack(
            [new_atoms.get_atomic_numbers(), adpart.get_atomic_numbers()]
        )
        nlist = self.nlist_prototype(  # type: ignore
            self.covalent_max * np.array([covalent_radii[c] for c in chemicl_numbers])
        )
        num_atoms = len(new_atoms)
        atomic_indices = list(range(num_atoms, num_atoms + len(adpart)))
        check_distance_func = functools.partial(
            check_atomic_distances_by_neighbour_list,
            neighlist=nlist,
            atomic_indices=atomic_indices,
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
            # max_attempts=self.MAX_RANDOM_ATTEMPTS,
            max_attempts=100,
            check_distance_func=check_distance_func,
            rng=rng,
        )

        _, _, state, num_attempts = info.split("_")
        if state == "success":
            self._print(f"succeed to insert after {num_attempts} attempts...")
            self._extra_info = f"Insert_{species}_{adpart_tag}"  # type: ignore
        elif state == "failure":
            self._print(f"failed to insert after {num_attempts} attempts...")
        else:
            raise Exception("This should not happen.")

        return new_atoms

    def _remove(
        self,
        atoms: Atoms,
        species: str,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> Atoms:
        """"""
        # We cannot use deepcopy here as ase does not delete some arrays,
        # for example, the forces.
        new_atoms = atoms.copy()

        # Pick one random particle
        species_indices = self._select_species(new_atoms, [species], rng)

        # The tags for atoms in the species should be the same,
        # we need check this?
        particle_tag = new_atoms.get_tags()[species_indices][0]

        # Remove then
        del new_atoms[species_indices]

        # Update info
        self._extra_info = f"Remove_{self.species}_{particle_tag}"  # type: ignore

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
        reservoir: dict,
        use_bias: bool = True,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(
            *args,
            **kwargs,
        )

        self.species = reservoir["species"]
        self.mu = reservoir["mu"]

        self.use_bias = use_bias

        self.nlist_prototype = functools.partial(
            NeighborList, skin=0.0, self_interaction=False, bothways=True
        )

        return

    def run(
        self, atoms: Atoms, rng: np.random.Generator = np.random.default_rng()
    ) -> Optional[Atoms]:
        """"""
        # Check species in the region
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

        # Choose a species to exchange
        # valid_species = [k for k, v in tag_dict.items() if len(v) > 0]
        assert isinstance(self._curr_tags_dict, dict)
        num_particles = len(self._curr_tags_dict.get(self.species, []))

        # Choose insert or remove
        if num_particles > 0:
            rn_ex = rng.uniform()
            if rn_ex < 0.5:
                self._print("...insert...")
                self._curr_operation = "insert"
                new_atoms = self._insert(atoms, self.species, rng)
            else:
                self._print("...remove...")
                self._curr_operation = "remove"
                new_atoms = self._remove(atoms, self.species, rng)
        else:
            self._print("...insert...")
            self._curr_operation = "insert"
            new_atoms = self._insert(atoms, self.species, rng)

        return new_atoms

    def metropolis(
        self,
        prev_ene: float,
        curr_ene: float,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> bool:
        """"""
        # - acceptance ratio
        kBT_eV = units.kB * self.temperature
        beta = 1.0 / kBT_eV  # 1/(kb*T), eV

        # - cubic thermo de broglie
        hplanck = units._hplanck  # J/Hz = kg*m2*s-1
        # _mass = np.sum([data.atomic_masses[data.atomic_numbers[e]] for e in expart]) # g/mol
        _species = convert_string_to_atoms(self.species)
        _species_mass = np.sum(_species.get_masses())
        # print("species mass: ", _mass)
        _mass = _species_mass * units._amu
        kbT_J = kBT_eV * units._e  # J = kg*m2*s-2
        cubic_wavelength = (
            hplanck / np.sqrt(2 * np.pi * _mass * kbT_J) * 1e10
        ) ** 3  # thermal de broglie wavelength

        # Compute coefficient
        # Determine number of exchangeable particles
        assert isinstance(self._curr_tags_dict, dict)
        if self.species not in self._curr_tags_dict:
            self._curr_tags_dict[self.species] = []
        nexatoms = len(self._curr_tags_dict[self.species])

        # Compute composed coefficient
        if self._curr_operation == "insert":
            assert isinstance(self._curr_volume, float)
            coef = self._curr_volume / (nexatoms + 1) / cubic_wavelength
            ene_diff = curr_ene - self.mu - prev_ene
        elif self._curr_operation == "remove":
            coef = nexatoms * cubic_wavelength / self._curr_volume
            ene_diff = curr_ene + self.mu - prev_ene
        else:
            raise RuntimeError(f"Unknown exchange operation {self._curr_operation}.")

        acc_ratio = np.min([1.0, coef * np.exp(-beta * (ene_diff))])

        content = "\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n" % (
            self._curr_volume,
            nexatoms,
            cubic_wavelength,
            coef,
        )
        # content = "\nVolume %.4f Beta %.4f Coefficient %.4f\n" %(
        #    0., beta, coef
        # )
        content += "Energy Difference %.4f [eV]\n" % ene_diff
        content += "Accept Ratio %.4f\n" % acc_ratio
        for x in content.split("\n"):
            self._print(x)

        rn_move = rng.uniform()
        self._print(f"{self.__class__.__name__} Probability %.4f" % rn_move)

        # - reset stored temp data
        self._curr_operation = None
        self._curr_tags_dict = None
        self._curr_volume = None

        return rn_move < acc_ratio

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += (
            f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        )
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"reservoir: "
        content += f"  species {self.species} with chemical potential {self.mu} [eV]\n"
        content += f"  within the region {self.region}\n"

        return content

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["reservoir"] = dict(species=self.species, mu=self.mu)
        params["use_bias"] = self.use_bias

        return params


if __name__ == "__main__":
    ...
