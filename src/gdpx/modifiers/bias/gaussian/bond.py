#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
import pathlib

from typing import Optional, List, Tuple

import numpy as np

from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList, natural_cutoffs


from ..utils import get_bond_information


def compute_bond_gaussian_energy_and_forces(
    vec_mic, dis, equ_dis, bstrain, saved_bstrains, sigma: float, omega: float
):
    """Compute bias energy and forces on a single bond strain."""
    # print(f"{bstrain =}")
    # print(f"{saved_bstrains =}")
    # - compute energy
    x, x_t = bstrain, saved_bstrains
    x1 = x - x_t
    x2 = x1**2 / 2.0 / sigma**2  # uniform sigma?
    v = omega * np.exp(-np.sum(x2, axis=1))

    energy = v.sum(axis=0)

    # - compute forces
    # -- dE/ds _ shape (1, cv_dim)
    dEds = np.sum(-v[:, np.newaxis] * x1 / sigma**2, axis=0)[np.newaxis, :]

    # -- ds/dc # cv gradient wrt coordinate
    dsdx = vec_mic / dis / equ_dis

    forces = np.sum(-dEds * dsdx, axis=0)

    return energy, forces


class BondGaussianCalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        bonds: List[str] = ["C", "H", "O"],
        cov_equi_ratio: float = 1.0,
        covalent_ratio: Tuple[float, float] = [0.8, 1.6],
        target_indices: Optional[List[int]] = None,
        sigma: float = 0.1,
        omega: float = -0.2,
        pace: int = 1,
        **kwargs,
    ):
        """"""
        super().__init__(**kwargs)

        self.sigma = sigma
        self.omega = omega

        self.pace = pace

        self.sequ = cov_equi_ratio
        self.smin, self.smax = covalent_ratio

        #: Equilibrium bond distance dict.
        if isinstance(bonds[0], str):
            symbols = bonds
            bonds = list(itertools.product(bonds, bonds))
        else:
            symbols_, bonds_ = [], []
            for i, j in bonds:
                symbols_.extend([i, j])
                bonds_.append((i, j))
                bonds_.append((j, i))
            symbols = list(set(symbols_))
            bonds = list(set(bonds_))
        self.symbols = symbols
        self.bonds = bonds

        radii = {s: cov_equi_ratio * covalent_radii[atomic_numbers[s]] for s in symbols}
        self.eqdis_dict = {k: radii[k[0]] + radii[k[1]] for k in bonds}

        #:
        if target_indices is not None:
            self.target_indices = target_indices
        else:
            self.target_indices = None

        # - private properties
        self.neighlist = None

        self._num_steps = 0

        self._initial_bonds = []

        self._reacted_bonds = []

        self._history_records = {}

        return

    @property
    def num_steps(self) -> int:
        """"""

        return self._num_steps

    def reset_metadata(self):
        """"""

        return

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=["positions", "numbers", "cell"],
    ):
        """"""
        super().calculate(atoms, properties, system_changes)

        if self.num_steps == 0:
            # - write bias log
            content = ""
            content += f"# {self.sequ} [{self.smin} {self.smax}]\n"
            content += f"# {self.eqdis_dict}\n"
            content += f"# pace {self.pace} width {self.sigma} height {self.omega}\n"
            content += "# {:>10s}  {:>12s}  {:>12s}\n".format(
                "step", "num_biased", "num_reacted"
            )
            log_fpath = pathlib.Path(self.directory) / "info.log"
            with open(log_fpath, "w") as fopen:
                fopen.write(content)

            # - write event log
            content = "# reaction events\n"
            event_fpath = pathlib.Path(self.directory) / "event.log"
            with open(event_fpath, "w") as fopen:
                fopen.write(content)

        # - create a neighlist
        if self.neighlist is None:
            self.neighlist = NeighborList(
                self.smax * self.sequ * np.array(natural_cutoffs(atoms)),
                skin=0.0,
                self_interaction=False,
                bothways=False,
            )
        else:
            ...
        self.neighlist.update(atoms)

        # - find species
        if self.target_indices is None:
            self.target_indices = [
                i for i, a in enumerate(atoms) if a.symbol in self.symbols
            ]
            print(f"{self.target_indices =}")

        # - get initial bonds
        if self.num_steps == 0:
            bond_pairs, _, _, _ = get_bond_information(
                atoms,
                self.neighlist,
                self.eqdis_dict,
                covalent_min=0.8,
                target_indices=self.target_indices,
                # allowed_bonds=[("C", "O"), ("C", "H"), ("H", "O")],
                allowed_bonds=self.bonds
            )
            self._initial_bonds = bond_pairs
            print(f"{self._initial_bonds =}")

        energy, forces = self._compute_bias(atoms)

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        self._write_step()

        self._num_steps += 1

        return

    def _compute_bias(self, atoms: Atoms):
        """"""
        # - find species
        bond_pairs, bond_distances, bond_shifts, equi_distances = get_bond_information(
            atoms,
            self.neighlist,
            self.eqdis_dict,
            covalent_min=self.smin,
            target_indices=self.target_indices,
            allowed_bonds=self.bonds,
        )
        # print(f"{bond_pairs =}")
        # print(f"{bond_distances =}")

        # - compute energy and forces
        num_bonds = len(bond_pairs)
        if num_bonds > 0:
            energy = 0.0
            forces = np.zeros((atoms.positions.shape))
            # -- add bond with the max strain to the reactive bond list
            bond_strains = (bond_distances - equi_distances) / equi_distances
            max_index = np.argmax(bond_strains)
            max_bond_pair = bond_pairs[max_index]
            if self.num_steps % self.pace == 0:
                if max_bond_pair not in self._initial_bonds:
                    if max_bond_pair in self._history_records:
                        self._history_records[max_bond_pair].append(
                            [bond_strains[max_index]]
                        )
                    else:
                        self._history_records[max_bond_pair] = [[bond_strains[max_index]]]
                    self._write_event(max_bond_pair, "biased")
            # print(f"{self._history_records =}")
            # -- apply gaussian bias on reactive bonds
            for i, bond_pair in enumerate(bond_pairs):
                if bond_pair in self._history_records:
                    b_i, b_j = bond_pair
                    bond_strain = bond_strains[i]
                    # check whether bond is formed and clear history if so
                    if bond_strain <= 0.05:
                        if bond_pair not in self._reacted_bonds:
                            self._reacted_bonds.append(bond_pair)
                            self._history_records[bond_pair] = []
                            self._write_event(bond_pair, "bond_formed")
                        continue
                    else:
                        # FIXME: thereacted bond becomes reactive again?
                        if bond_pair not in self._reacted_bonds:
                            ...
                        else:
                            if bond_strain > 0.2:
                                self._history_records[bond_pair].append([bond_strain])
                                self._write_event(bond_pair, "bond_broken")
                            else:
                                continue
                    vec_mic = atoms.positions[b_i] - (
                        atoms.positions[b_j] + bond_shifts[i]
                    )
                    saved_bstrains = np.array(self._history_records[bond_pair])
                    curr_energy, curr_forces = compute_bond_gaussian_energy_and_forces(
                        vec_mic,
                        bond_distances[i],
                        equi_distances[i],
                        bstrain=bond_strain,
                        saved_bstrains=saved_bstrains,
                        sigma=self.sigma,
                        omega=self.omega,
                    )
                    energy += curr_energy
                    forces[b_i] += curr_forces
                    forces[b_j] -= curr_forces
                    # print(f"{b_i} ~ {b_j}: {curr_forces =}")
        else:
            energy = 0.0
            forces = np.zeros((atoms.positions.shape))

        return energy, forces

    def _write_step(self):
        """"""
        num_biased = len(self._history_records)
        num_reacted = len(self._reacted_bonds)

        content = ""
        content += "{:>12d}  {:>12d}  {:>12d}\n".format(
            self.num_steps, num_biased, num_reacted
        )

        log_fpath = pathlib.Path(self.directory) / "info.log"
        with open(log_fpath, "a") as fopen:
            fopen.write(content)

        return
    
    def _write_event(self, bond_pair: Tuple[int, int], event: str):
        """"""
        pair_name = "_".join([str(x) for x in bond_pair])

        content = f"{self.num_steps:>8d}  {pair_name:>24s}  {event:>24s}\n"
        event_fpath = pathlib.Path(self.directory) / "event.log"
        with open(event_fpath, "a") as fopen:
            fopen.write(content)

        return


if __name__ == "__main__":
    ...
