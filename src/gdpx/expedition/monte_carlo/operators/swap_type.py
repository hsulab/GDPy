#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.data import chemical_symbols

from .operator import BaseMCOperator


class SwapTypeOperator(BaseMCOperator):
    """Monte Carlo operator for swapping particle types.

    NOTE:
        This operator supports only changing single atoms, not molecules.

    """

    name: str = "swap_type"

    def __init__(
        self,
        particles: list[str],
        chempots: list[float],
        *args,
        **kwargs,
    ) -> None:
        """Initialise a MC swap type operator.

        Args:
            particles: The particles that can swap type.

        """
        super().__init__(
            *args,
            **kwargs,
        )

        self.particles = particles
        if len(self.particles) < 2:
            raise Exception("At least two particles are required for swap type operator.")

        for p in self.particles:
            if p not in chemical_symbols:
                raise Exception(f"Particle {p} is not a valid chemical symbol.")

        self.chempots = chempots
        if len(self.particles) != len(self.chempots):
            raise Exception("Number of particles and chemical potentials must match.")

        # Some state information after mc attempts and before energy evaluation
        self._state = {}

        return

    @property
    def state(self) -> dict:
        """Get the state of the operator."""
        return self._state

    def run(self, atoms: Atoms, rng=np.random.default_rng()) -> Optional[Atoms]:
        """"""
        # Check particles in the region
        super().run(atoms)
        self._extra_info = "-"

        # We need covalent bond distances for neighbour check
        assert hasattr(self, "bond_distance_dict")

        # Get a new copy of the atoms
        new_atoms = copy.deepcopy(atoms)

        # Find two particle types that can swap, particles of the first type should exist
        ptypes_in_region = set(self._curr_tags_dict.keys()) & set(self.particles)
        num_ptypes_in_region = len(ptypes_in_region)
        if num_ptypes_in_region < 1:
            # Skip if no particles in the region and try later if other operators such as
            # exchange can insert particles
            return None

        num_particles = len(self.particles)

        first_ptype_index = rng.choice(num_ptypes_in_region, 1)[0]
        for _ in range(100):
            second_ptype_index = rng.choice(num_particles, 1)[0]
            if second_ptype_index != first_ptype_index:
                break
        else:
            # If we cannot find a second type, we cannot swap
            self._print("Cannot find a second particle type to swap with.")
            return None

        first_ptype = self.particles[first_ptype_index]
        second_ptype = self.particles[second_ptype_index]

        # Change selected particles to another type based on chemical potential difference
        self._print(self.indent + "--> mcattempt")
        for i in range(self.MAX_RANDOM_ATTEMPTS):
            # Pick an atom either index of an atom or tag of an moiety
            pick_one = self._select_species(new_atoms, [first_ptype], rng=rng)
            assert len(pick_one) == 1, "Only one atom should be selected for swap type operator."
            new_atoms[pick_one[0]].symbol = second_ptype
            # TODO: renormalise other properties such as velocity, charge, and magnetic moment
            self._print(self.indent + f"succeed to random after {i+1} attempts...")
            self._extra_info = f"ST_{first_ptype}_{second_ptype}_{pick_one[0]}"
            # Compute concentration
            num_first_ptype = len(self._curr_tags_dict.get(first_ptype, []))
            num_second_ptype = len(self._curr_tags_dict.get(second_ptype, []))
            prev_x = num_first_ptype / (num_first_ptype + num_second_ptype)
            curr_x = (num_first_ptype - 1) / (num_first_ptype + num_second_ptype)
            self._state = {
                "first_ptype": first_ptype,
                "second_ptype": second_ptype,
                "picked_atom_index": pick_one[0],
                "num_first_ptype": num_first_ptype - 1,
                "num_second_ptype": num_second_ptype + 1,
                "dX": curr_x - prev_x,
            }
            break
        else:
            new_atoms = None
            self._extra_info = f"SwapType_Failed"

        return new_atoms

    def metropolis(self, prev_ene: float, curr_ene: float, rng=np.random.default_rng()) -> bool:
        """Metropolis criterion for the swap type operator."""
        # Temperature parameters
        kBT_eV = units.kB * self.temperature
        beta = 1.0 / kBT_eV  # 1/(kb*T), eV

        # Check number of particles in the region
        assert isinstance(self._curr_tags_dict, dict)

        first_ptype: str = self._state.get("first_ptype")
        num_first_ptype: int = self._state.get("num_first_ptype")

        second_ptype: str = self._state.get("second_ptype")
        num_second_ptype: int = self._state.get("num_second_ptype")

        mu_diff = self.chempots[self.particles.index(first_ptype)] - self.chempots[self.particles.index(second_ptype)]

        # Energetic parameters
        ene_diff = curr_ene - prev_ene
        ene_sgce = ene_diff + mu_diff

        # Compute the prefactor
        prefactor = 1.0
        region_volume = self.region.get_volume()

        # Propability of acceptance
        acc_ratio = np.min([1.0, prefactor * np.exp(-beta * ene_sgce)])
        ran_ratio = rng.uniform()

        # Some log information
        content = "--> mcstate\n"
        content += f"Volume {region_volume:>12.4f} [A^3] Beta {beta:>12.4f} [1/eV]\n"
        content += f"Prefactor {prefactor:>12.4f}\n"
        content += f"Particle One: {first_ptype:<4s} ({num_first_ptype})\n"
        content += f"Particle Two: {second_ptype:<4s} ({num_second_ptype})\n"
        content += f"dMu {mu_diff:>11.4f} [eV]\n"
        content += f"dE {ene_diff:>12.4f} [eV]  " + f"dF {ene_sgce:>12.4f} [eV]\n"
        content += f"Accept {acc_ratio:>4.2e} >? {ran_ratio:>4.2e}"
        for x in content.split("\n"):
            self._print(self.indent + x)

        # Clear the state information
        self._state = {}

        return ran_ratio < acc_ratio

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

        # add indent
        content = self.indent + content.replace("\n", "\n" + self.indent)

        return content


if __name__ == "__main__":
    ...
