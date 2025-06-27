#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import copy
from typing import Callable, Optional

import numpy as np
from ase import Atoms, data, units

from gdpx.core.register import registers


class BaseMCOperator(abc.ABC):

    #: Operator name.
    name: str = "abstract"

    #: Maximum attempts to generate the new position of a species.
    MAX_RANDOM_ATTEMPTS = 1000

    #: Print function.
    _print: Callable = print

    #: Debug function.
    _debug: Callable = print

    def __init__(
        self,
        region: dict = {},
        group: Optional[str] = None,
        temperature: float = 300.0,
        pressure: float = 1.0,
        covalent_ratio=[0.8, 2.0],
        allow_isolated: bool = False,
        skip_distance_check: bool = False,
        use_rotation=True,
        prob: float = 1.0,
    ) -> None:
        """Initialise the modification operator.

        Args:
            temperature: Monte Carlo temperature [K].
            pressure: Monte Carlo pressure [bar].
            covalent_ratio: Minimum and maximum percentages of covalent distance.

        """
        super().__init__()

        # Propability to be selected
        self.prob = prob

        # Check region
        region_params = copy.deepcopy(region)
        region_method = region_params.pop("method", "auto")
        self.region = registers.create("region", region_method, convert_name=True, **region_params)

        # Check group
        self.group = group

        # Check thermodynamic parameters
        self.temperature = temperature
        self.pressure = pressure

        # Neighour settings
        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        self.allow_isolated = allow_isolated

        self.skip_distance_check = skip_distance_check

        # Molecular rotation
        self.use_rotation = use_rotation

        # Some state information after mc attempts and before energy evaluation
        self._atoms = None
        self._state = {}

        # Indent before any print or string
        self._indent = ""

        return

    @property
    def state(self) -> dict:
        """Get the state of the operator."""
        return self._state

    @property
    def indent(self) -> str:
        """Get the indent string."""
        return self._indent

    @indent.setter
    def indent(self, indent: str) -> None:
        """Set the indent string."""
        self._indent = indent

        return

    def _check_region(self, atoms: Atoms) -> None:
        """Set _curr_tags_dict."""
        if self.region.__class__.__name__ == "AutoRegion":
            self.region._atoms = atoms

        tags_dict = self.region.get_tags_dict(atoms)
        content = "particles in system:  "
        content += "  " + "  ".join([str(k) + " " + str(len(v)) for k, v in tags_dict.items()])
        for x in content.split("\n"):
            self._print(self.indent + x)

        self._curr_tags_dict = self.region.get_contained_tags_dict(atoms, tags_dict)
        content = "          in region:  "
        content += "  " + "  ".join([str(k) + " " + str(len(v)) for k, v in self._curr_tags_dict.items()])
        for x in content.split("\n"):
            self._print(self.indent + x)

        return

    def _select_species(
        self, atoms: Atoms, particles: Optional[list[str]] = None, rng=np.random.default_rng()
    ) -> list[int]:
        """"""
        # - pick a particle (atom/molecule)
        tags_within_region = []
        for k, v in self._curr_tags_dict.items():
            if particles is not None and k not in particles:
                continue
            tags_within_region.extend(v)
        self._print(self.indent + f"ntags of {particles}: {len(tags_within_region)}")

        if len(tags_within_region) > 0:
            picked_tag = rng.choice(tags_within_region)
            tags = atoms.get_tags()
            species_indices = [i for i, t in enumerate(tags) if t == picked_tag]
            self._print(
                self.indent + f"selected tag: {picked_tag} species: {atoms[species_indices].get_chemical_formula()}"
            )
        else:
            picked_tag = None
            species_indices = None
            raise RuntimeError(f"{self.__class__.__name__} does not have {particles}.")

        return species_indices

    def _rotate_species(self, species: Atoms, rng=np.random):
        """"""
        species_ = species.copy()  # TODO: make clean atoms?
        org_com = np.mean(species_.positions, axis=0)
        if self.use_rotation and len(species_) > 1:
            phi, theta, psi = 360 * rng.uniform(0, 1, 3)
            species_.euler_rotate(phi=phi, theta=0.5 * theta, psi=psi, center=org_com)

        return species_

    def check_overlap_neighbour(self, nl, new_atoms: Atoms, cell, species_indices: list[int]) -> bool:
        """Check whether the species position is valid.

        Use neighbour list to check newly added atom is neither too close or too
        far from other atoms. The neighbour list is based on covalent_max distance.
        We have three status, `valid`, `invalid`, adn `isolated`.
        The situations being considered invalid:
            - Atomic distance too close (covalent_min).
            - All atoms in the species are isolated from the rest of the system.

        """
        assert self.blmin is not None, "BondLengthMinimumDict is not properly set."

        # -
        num_atoms_in_species = len(species_indices)

        species_status = ["valid"] * num_atoms_in_species
        self._print(f"- {species_indices =}")

        # - get symbols here since some operators may change the symbol
        chemical_symbols = new_atoms.get_chemical_symbols()

        nl.update(new_atoms)
        for iatom, idx_pick in enumerate(species_indices):
            indices, offsets = nl.get_neighbors(idx_pick)
            self._debug(f"  check index {idx_pick} {new_atoms.positions[idx_pick]} nneighs: {len(indices)}")
            if len(indices) > 0:
                # --
                if all([(ni in species_indices) for ni in indices]):
                    species_status[iatom] = "isolated"
                    continue
                # -- check inter-species atomic distances
                for ni, offset in zip(indices, offsets):
                    # NOTE: Check if the species contact other atoms
                    #       in a reasonable distance.
                    #       Intra-species distance will not be checked.
                    if ni not in species_indices:
                        dis = np.linalg.norm(
                            new_atoms.positions[idx_pick] - (new_atoms.positions[ni] + np.dot(offset, cell))
                        )
                        pairs = [chemical_symbols[ni], chemical_symbols[idx_pick]]
                        pairs = tuple([data.atomic_numbers[p] for p in pairs])
                        if dis <= self.blmin[pairs]:
                            species_status[iatom] = "invalid"
                            self._debug(f"  distance: {ni} {dis} {self.blmin[pairs]}")
                            break
                    else:
                        ...
                else:
                    species_status[iatom] = "valid"
            else:
                # We need BREAK here?
                # For a single-atom species, one loop finishes.
                # For a multi-atom species, the code will not happen except when
                # the covalent_max is WAY TOO SMALL then ...
                # assert num_atoms_in_species == 1,
                species_status[iatom] = "isolated"
        self._print(f"  {species_status =}")

        status_ = []
        for s in species_status:
            if s == "valid" or s == "invalid":
                ...
            else:  # isolated
                if self.allow_isolated:
                    s = "valid"
                else:
                    s = "invalid"
            status_.append(s)

        return any([s == "invalid" for s in status_])

    @abc.abstractmethod
    def run(self, atoms: Atoms, rng=np.random.default_rng()) -> Optional[Atoms]:
        """Modify the input atoms.

        Returns:
            A new atoms is returned if the modification succeeds otherwise None is
            returned.

        """
        self._check_region(atoms)

        ...

    @abc.abstractmethod
    def metropolis(self, prev_ene: float, curr_ene: float, rng: np.random.Generator = np.random.default_rng()) -> bool:
        """Monte Carlo."""

        ...

    def as_dict(self) -> dict:
        """"""
        params = {}
        params["method"] = self.name
        params["region"] = self.region.as_dict()
        params["group"] = self.group
        params["temperature"] = self.temperature
        params["pressure"] = self.pressure
        params["use_rotation"] = self.use_rotation
        params["covalent_ratio"] = [self.covalent_min, self.covalent_max]
        params["allow_isolated"] = self.allow_isolated
        params["skip_distance_check"] = self.skip_distance_check
        params["prob"] = self.prob

        params = copy.deepcopy(params)

        return params


def metropolis_by_energy_difference(
    prev_ene: float,
    curr_ene: float,
    temperature: float,
    region,
    rng: np.random.Generator = np.random.default_rng(),
    indent: str = "",
    print_func: Callable = print,
) -> bool:
    """"""
    # Temperature parameters
    kBT_eV = units.kB * temperature
    beta = 1.0 / kBT_eV  # 1/(kb*T), eV

    # Compute the prefactor
    prefactor = 1.0
    region_volume = region.get_volume()

    # Propability of acceptance
    ene_diff = curr_ene - prev_ene
    acc_ratio = np.min([1.0, prefactor * np.exp(-beta * (ene_diff))])
    ran_ratio = rng.uniform()

    # Some log information
    content = "--> mcstate\n"
    content += f"Volume {region_volume:>12.4f} [A^3] Beta {beta:>12.4f} [1/eV]\n"
    content += f"Prefactor {prefactor:>12.4f}\n"
    content += f"dE {ene_diff:>12.4f} [eV]\n"
    content += f"Accept {acc_ratio:>4.2e} >? {ran_ratio:>4.2e}"
    for s in content.split("\n"):
        print_func(indent + s)

    # Clear the state information

    return ran_ratio < acc_ratio


if __name__ == "__main__":
    ...
