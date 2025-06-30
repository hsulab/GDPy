#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.data import chemical_symbols, covalent_radii
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import NeighborList

from gdpx.geometry.particle import translate_then_rotate

from .basic import BasicExchangeOperator


def find_cavity_points(
    atoms: Atoms,
    atomic_indices: list[int],
    nlist: NeighborList,
    bond_distance_dict: dict,
    covalent_min: float = 0.8,
    covalent_max: float = 2.0,
    cav_dmin: Optional[float] = None,
    cav_dmax: Optional[float] = None,
    check_isolated: bool = True,
) -> list[int]:
    """Find cavity points in the given atoms by neighbour list.

    Args:
        atoms: The Atoms object with the original atoms plus trial points.
        atomic_indices: The indices of the trial points to check for cavities.
        nlist: The neighbour list for the atoms.
        bond_distance_dict: A dictionary mapping atomic pairs to their bond distances.
        covalent_min: The minimum covalent ratio for distance checks.
        covalent_max: The maximum covalent ratio for distance checks.
        cav_dmin: Optional minimum distance for cavity points.
        cav_dmax: Optional maximum distance for cavity points.
        check_isolated: Whether to check if the cavity points are isolated.

    Returns:
        A list of indices of the cavity points found in the atoms.

    """
    chemical_numbers = atoms.get_atomic_numbers()

    # Find cavity points
    cell = atoms.get_cell()
    cavity_indices = []
    for i in atomic_indices:
        n_indices, n_offsets = nlist.get_neighbors(i)
        found_isolated = True
        for j, o in zip(n_indices, n_offsets):
            if j not in atomic_indices:  # skip trial particles
                distance = np.linalg.norm(atoms.positions[i] - (atoms.positions[j] + np.dot(o, cell)))
                if cav_dmin is not None:
                    dmin = cav_dmin
                    dmax = cav_dmax
                else:
                    atomic_pair = (chemical_numbers[i], chemical_numbers[j])
                    dmin: float = bond_distance_dict[atomic_pair] * covalent_min
                    dmax: float = bond_distance_dict[atomic_pair] * covalent_max

                if distance <= dmin:
                    break  # too close to an existing atom
                else:
                    if check_isolated:
                        if distance <= dmax:
                            found_isolated = False
                        else:
                            ...
                    else:
                        ...
            else:
                ...
        else:
            if not check_isolated:
                cavity_indices.append(i)  # no close atoms found
            else:
                if not found_isolated:
                    cavity_indices.append(i)
                else:
                    ...

    return cavity_indices


class CavityExchangeOperator(BasicExchangeOperator):
    """
    Cavity exchange operator for Monte Carlo simulations.

    This operator allows the insertion of particles into a cavity defined by a region.
    It checks the distances between atoms to ensure that the new particle does not
    violate any distance constraints.
    """

    MIN_RANDOM_TAG: int = 10_000

    MAX_RANDOM_TAG: int = 100_000

    def __init__(
        self, num_trials: int, cavity_distance: Optional[tuple[float, Optional[float]]] = None, *args, **kwargs
    ):
        """"""
        super().__init__(*args, **kwargs)

        if self.particles[0] not in chemical_symbols:
            raise Exception("CavityExchangeOperator only supports single-atom particles.")

        self.num_trials = num_trials

        # The distance criteria for finding cavities.
        # If cavity_distance is not provided, we use covalent_ratio.
        if cavity_distance is not None:
            # The minimum distance must be provided and the maximum distance is optional.
            if len(cavity_distance) != 2 or cavity_distance[0] <= 0:
                raise Exception("cavity_distance must be a tuple of (min_distance, max_distance).")
        self.cavity_distance = cavity_distance

        return

    def _update_volume(self, atoms: Atoms) -> float:
        """Update the volume of the region based on the current atoms."""

        acc_volume = self.region.get_volume()

        return acc_volume

    def _insert(
        self, atoms: Atoms, particle: str, particle_instance: Atoms, rng: np.random.Generator = np.random.default_rng()
    ) -> Atoms:
        """Insert a particle into the cavity."""
        assert hasattr(self, "bond_distance_dict"), "Bond distance dictionary must be set."

        # Get reference
        self._atoms = atoms
        new_atoms = atoms

        # Prepare particle to add
        adpart = particle_instance.copy()

        # Add velocity in case the mixed MC/MD is performed
        MaxwellBoltzmannDistribution(adpart, temperature_K=self.temperature, rng=rng)

        # Choose a tag for the particle
        used_tags = set(atoms.get_tags().tolist())
        adpart_tag = 0
        while adpart_tag in used_tags:
            adpart_tag = rng.integers(self.MIN_RANDOM_TAG, self.MAX_RANDOM_TAG)
        adpart_tag = int(adpart_tag)
        self._print(self.indent + f"adpart {adpart.get_chemical_formula()} tag: {adpart_tag} {type(adpart_tag)}")

        # Add all trial particles
        num_atoms = len(new_atoms)
        num_atoms_in_particle = len(adpart)

        atomic_indices = list(range(num_atoms, num_atoms + num_atoms_in_particle * self.num_trials))
        self._state["atomic_indices"] = atomic_indices[:num_atoms_in_particle]

        random_positions = self.region.get_random_positions(size=self.num_trials, rng=rng)
        for random_position in random_positions:
            trial_adpart = adpart.copy()
            trial_adpart = translate_then_rotate(trial_adpart, position=random_position, use_com=True, rng=rng)
            new_atoms.extend(trial_adpart)

        # Use neighbour list to find cavity points
        num_atoms_total = len(new_atoms)

        check_isolated = True
        cav_dmin, cav_dmax = None, None
        if self.cavity_distance is not None:
            cutoffs = self.cavity_distance[0] / 2.0 * np.ones(num_atoms_total, dtype=float)
            cav_dmin, cav_dmax = self.cavity_distance
            if cav_dmax is None:
                check_isolated = False
        else:
            cutoffs = self.covalent_max * np.array([covalent_radii[c] for c in new_atoms.get_atomic_numbers()])

        nlist = self.nlist_prototype(cutoffs)
        nlist.update(new_atoms)

        cavity_indices = find_cavity_points(
            atoms=new_atoms,
            atomic_indices=atomic_indices,
            nlist=nlist,
            bond_distance_dict=self.bond_distance_dict,  # type: ignore
            covalent_min=self.covalent_min,
            covalent_max=self.covalent_max,
            cav_dmin=cav_dmin,
            cav_dmax=cav_dmax,
            check_isolated=check_isolated,
        )

        num_cavities = len(cavity_indices)
        if num_cavities == 0:
            symm_factor = 1.0
            selected_position = random_positions[-1]
        else:
            symm_factor = self.num_trials / num_cavities
            selected_position = random_positions[cavity_indices[0] - num_atoms]
        self._state["symm_factor"] = symm_factor

        del new_atoms[atomic_indices[1:]]  # remove trial particles
        new_atoms.positions[atomic_indices[0]] = selected_position

        self._extra_info = f"Insert_{particle}_{adpart_tag}_c{num_cavities}"

        return new_atoms

    def _remove(self, atoms: Atoms, particle: str, rng: np.random.Generator = np.random.default_rng()) -> Atoms:
        """Remove a particle randomly."""
        self._atoms = atoms
        new_atoms = atoms

        # Pick one random particle
        removed_indices = self._select_species(new_atoms, [particle], rng)

        removed_particle = new_atoms[removed_indices]
        assert isinstance(removed_particle, Atoms), "Removed particle should be an Atoms object."
        tags = removed_particle.get_tags()
        assert len(set(tags)) == 1, "All tags for the selected atoms should be the same."
        particle_tag = tags[0]

        self._state["removed_particle"] = removed_particle

        # Check cavity
        num_atoms = len(new_atoms)
        num_atoms_in_particle = len(removed_particle)

        atomic_indices = removed_indices + list(
            range(num_atoms, num_atoms + num_atoms_in_particle * (self.num_trials - 1))
        )

        # Prepare particle to add
        adpart = self._particle_instances[self.particles.index(particle)].copy()

        random_positions = self.region.get_random_positions(size=self.num_trials - 1, rng=rng)
        for random_position in random_positions:
            trial_adpart = adpart.copy()
            trial_adpart = translate_then_rotate(trial_adpart, position=random_position, use_com=True, rng=rng)
            new_atoms.extend(trial_adpart)

        # Use neighbour list to find cavity points
        num_atoms_total = len(new_atoms)

        check_isolated = True
        cav_dmin, cav_dmax = None, None
        if self.cavity_distance is not None:
            cutoffs = self.cavity_distance[0] / 2.0 * np.ones(num_atoms_total, dtype=float)
            cav_dmin, cav_dmax = self.cavity_distance
            if cav_dmax is None:
                check_isolated = False
        else:
            cutoffs = self.covalent_max * np.array([covalent_radii[c] for c in new_atoms.get_atomic_numbers()])

        nlist = self.nlist_prototype(cutoffs)
        nlist.update(new_atoms)

        cavity_indices = find_cavity_points(
            atoms=new_atoms,
            atomic_indices=atomic_indices,
            nlist=nlist,
            bond_distance_dict=self.bond_distance_dict,  # type: ignore
            covalent_min=self.covalent_min,
            covalent_max=self.covalent_max,
            cav_dmin=cav_dmin,
            cav_dmax=cav_dmax,
            check_isolated=check_isolated,
        )

        num_cavities = len(cavity_indices)
        if num_cavities == 0:
            symm_factor = 1.0
        else:
            if removed_indices[0] in cavity_indices:
                symm_factor = self.num_trials / num_cavities
            else:
                symm_factor = 0.0
        self._state["symm_factor"] = symm_factor

        # Remove trial particles
        del new_atoms[atomic_indices[1:]]

        # Remove the selected particle
        del new_atoms[removed_indices]

        # Update info
        self._extra_info = f"Remove_{particle}_{particle_tag}_c{num_cavities}"  # type: ignore

        return new_atoms

    def revert_state(self, atoms: Atoms) -> None:
        """Revert the state of the atoms to the previous state."""
        operation = self._state.get("operation")
        if operation == "insert":
            atomic_indices = self._state.get("atomic_indices")
            del atoms[atomic_indices]
        elif operation == "remove":
            # The removed particle will be added to the end of the atoms,
            # the order of atoms has changed but the tags are preserved.
            removed_particle = self._state.get("removed_particle")
            atoms.extend(removed_particle)
        else:
            raise Exception(f"Unknown operation: {operation}")

        return

    def metropolis(self, prev_ene: float, curr_ene: float, rng: np.random.Generator = np.random.default_rng()) -> bool:
        """"""
        # Temperature parameters
        kBT_eV = units.kB * self.temperature
        beta = 1.0 / kBT_eV  # 1/(kb*T), eV

        # Get particle properties
        chempot = self.chempots[0]
        cubic_wavelength = self._cubic_wavelengths[0]

        # Compute the prefactor
        num_particles = self._state["num_particles"]
        region_volume = self._state["volume"]
        symm_factor = self._state["symm_factor"]

        ene_diff = curr_ene - prev_ene
        if self._state["operation"] == "insert":
            assert isinstance(region_volume, float)
            prefactor = region_volume / (num_particles + 1) / cubic_wavelength / symm_factor
            ene_gcmc = ene_diff - chempot
        elif self._state["operation"] == "remove":
            prefactor = symm_factor * num_particles * cubic_wavelength / region_volume
            ene_gcmc = ene_diff + chempot
        else:
            raise RuntimeError(f"Unknown exchange operation {self._state['operation']}.")

        acc_ratio = np.min([1.0, prefactor * np.exp(-beta * (ene_gcmc))])
        ran_ratio = rng.uniform()

        # Some log information
        content = "--> mcstate\n"
        content += f"Volume {region_volume:>12.4f} [A^3] Beta {beta:>12.4f} [1/eV]\n"
        content += f"Prefactor {prefactor:>12.4f} " + f"SymmFactor {symm_factor:>12.4f}\n"
        content += f"CubicWavelength {cubic_wavelength:>12.4f} [A^3]\n"
        content += f"N  {num_particles:>12d}\n"
        content += f"dE {ene_diff:>12.4f} [eV]  " + f"dF {ene_gcmc:>12.4f} [eV]\n"
        content += f"Accept {acc_ratio:>4.2e} >? {ran_ratio:>4.2e}"
        for x in content.split("\n"):
            self._print(self.indent + x)

        success = ran_ratio < acc_ratio
        if not success:
            assert self._atoms is not None, "Atoms should not be None when reverting state."
            self.revert_state(self._atoms)
        else:
            ...

        self._state = {}
        self._atoms = None

        return success

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["num_trials"] = self.num_trials
        params["cavity_distance"] = self.cavity_distance

        return params

    def __repr__(self) -> str:
        """"""
        content = super().__repr__()
        content += f"num_trials {self.num_trials}\n"
        content += f"cavity_distance {self.cavity_distance}\n"

        return content


if __name__ == "__main__":
    ...
