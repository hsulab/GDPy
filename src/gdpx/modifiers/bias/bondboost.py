#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import pathlib

from typing import Optional, List, Tuple

import numpy as np

from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList, natural_cutoffs

from .utils import get_bond_information


def compute_boost_energy_and_forces(
    positions,
    distances,
    shifts,
    ref_distances,
    bond_pairs,
    bond_strains,
    max_index: int,
    vmax=0.5,
    smax=0.5,
    curv: float = 0.98,
):
    """Compute bond-boost energy.

    u  = (1-(eps_max/smax)^2)
    dU = -2*(eps_max/smax^2)
    v  = (1-curv^2*(eps_max/smax)^2)
    dV = -2*curv^2*eps_max/smax^2

    The envelope function:
        A = u * u / v

    Args:
        ...

    Examples:
        >>>

    """
    num_bonds = len(bond_pairs)
    num_atoms = positions.shape[0]

    # - compute energy
    # V_b, shape (num_bonds, )
    vboost = vmax / num_bonds * (1 - (bond_strains / smax) ** 2)

    max_strain_ratio = bond_strains[max_index] / smax
    u = 1 - max_strain_ratio**2
    v = 1 - curv**2 * max_strain_ratio**2

    env = u * u / v
    energy = np.sum(vboost * env)

    # - compute forces
    forces = np.zeros((num_atoms, 3))

    # dV_b/deps_i, shape (num_bonds, )
    d_vboost = -vmax / num_bonds * 2 * bond_strains / smax**2

    # -- shared terms
    for p, (i, j) in enumerate(bond_pairs):
        frc_ij = (
            -env
            * d_vboost[p]
            * (positions[i] - (positions[j] + shifts[p]))
            / distances[p]
            / ref_distances[p]
        )
        forces[i] += frc_ij
        forces[j] += -frc_ij

    # -- the extra term for max_index
    du = -2 * max_strain_ratio / smax
    dv = -2 * curv**2 * max_strain_ratio / smax

    denv = du * (u / v) + u * (du * v - u * dv) / v**2

    max_i, max_j = bond_pairs[max_index]
    max_frc_ij = (
        -denv
        * (positions[max_i] - positions[max_j])
        / distances[max_index]
        / ref_distances[max_index]
        * np.sum(vboost)
    )
    forces[max_i] += max_frc_ij
    forces[max_j] += -max_frc_ij

    return energy, forces


class BondBoostCalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        curv: float = 0.98,
        vmax: float = 0.5,
        smax: float = 0.5,
        bonds: List[str] = ["C", "H", "O"],
        cov_equi_ratio: float = 1.0,
        covalent_ratio: Tuple[float, float] = [0.8, 1.6],
        target_indices: Optional[List[int]] = None,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        # bond-boost params
        #:V_max, eV
        self.vmax = vmax

        #: control the curvature near the boundary
        self.curv = curv

        #: q, maximum bond change compared to the reference state
        self.smax = smax

        self.covalent_ratio = covalent_ratio
        if not np.isclose(self.covalent_ratio[1], 1.0 + self.smax):
            self.covalent_ratio[1] = 1.0 + self.smax

        # aux params
        #: NeighborList
        self.neighlist = None

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

        radii = {s: cov_equi_ratio*covalent_radii[atomic_numbers[s]] for s in symbols}
        self.eqdis_dict = {k: radii[k[0]] + radii[k[1]] for k in bonds}

        #:
        if target_indices is not None:
            self.target_indices = target_indices
        else:
            self.target_indices = None

        return

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties=["energy"],
        system_changes=["positions"],
    ):
        """"""
        super().calculate(atoms, properties, system_changes)

        log_fpath = pathlib.Path(self.directory) / "info.log"
        if not log_fpath.exists():
            content = f"# {self.vmax =} {self.smax =} {self.curv = }\n"
            content += f"# {self.covalent_ratio[0]} {self.covalent_ratio[1]}\n"
            content += f"# {self.eqdis_dict}"
            content += f"# num_bonds max_i max_j max_strain\n"
            with open(log_fpath, "w") as fopen:
                fopen.write(content)

        if self.neighlist is None:
            covalent_max = self.covalent_ratio[
                1
            ]  # (1+smax) times covalent bond distance
            self.neighlist = NeighborList(
                covalent_max * np.array(natural_cutoffs(atoms)),
                skin=0.0,
                self_interaction=False,
                bothways=False,
            )
        else:
            ...
        self.neighlist.update(atoms)

        # - find species
        if self.target_indices is None:
            self.target_indices = [i for i, a in enumerate(atoms) if a.symbol in self.symbols]

        # - find bonds to boost
        bond_pairs, bond_distances, bond_shifts, equi_distances = get_bond_information(
            atoms,
            self.neighlist,
            self.eqdis_dict,
            covalent_min=self.covalent_ratio[0],
            target_indices=self.target_indices,
            allowed_bonds=self.bonds,
        )

        # - compute properties
        num_bonds = len(bond_pairs)
        if num_bonds > 0:
            # - compute properties
            bond_strains = (bond_distances - equi_distances) / equi_distances
            max_index = np.argmax(bond_strains)

            energy, forces = compute_boost_energy_and_forces(
                atoms.positions,
                bond_distances,
                bond_shifts,
                equi_distances,
                bond_pairs,
                bond_strains,
                max_index,
                vmax=self.vmax,
                smax=self.smax,
                curv=self.curv,
            )
            # print(f"{max_index =}")
            # print(f"{bond_pairs[max_index] =}")
            # print(f"{bond_distances[max_index] =}")
            # print(f"{bond_strains[max_index] =}")
            self._write_step(
                num_bonds,
                [atoms[x].symbol for x in bond_pairs[max_index]],
                bond_pairs[max_index],
                bond_distances[max_index],
                bond_strains[max_index],
            )
        else:
            energy = 0.0
            forces = np.zeros((atoms.positions.shape))
            self._write_step(num_bonds, [np.nan, np.nan], [np.nan, np.nan], np.nan, np.nan)

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        return

    def _write_step(
        self,
        num_bonds: int,
        bond_symb: Tuple[str, str],
        bond_pair: Tuple[int, int],
        distance: float,
        strain: float,
    ):
        """"""
        pair = "-".join([str(s) + "_" + str(x) for s, x in zip(bond_symb, bond_pair)])
        content = f"{num_bonds:>12d}  {pair:>24s}  {distance:>12.4f}  {strain:>12.4f}\n"

        log_fpath = pathlib.Path(self.directory) / "info.log"
        with open(log_fpath, "a") as fopen:
            fopen.write(content)

        return


if __name__ == "__main__":
    ...
