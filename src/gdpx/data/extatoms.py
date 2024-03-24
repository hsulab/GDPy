#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from ase import Atoms


class ScfErrAtoms(Atoms):

    def __init__(
        self,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        momenta=None,
        masses=None,
        magmoms=None,
        charges=None,
        scaled_positions=None,
        cell=None,
        pbc=None,
        celldisp=None,
        constraint=None,
        calculator=None,
        info=None,
        velocities=None,
    ):
        super().__init__(
            symbols,
            positions,
            numbers,
            tags,
            momenta,
            masses,
            magmoms,
            charges,
            scaled_positions,
            cell,
            pbc,
            celldisp,
            constraint,
            calculator,
            info,
            velocities,
        )

        self.info["scf_error"] = True

        return

    @staticmethod
    def from_atoms(atoms: Atoms) -> "ScfErrAtoms":
        """"""
        scferr_atoms = ScfErrAtoms(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            # tags=atoms.get_tags(),
            # magmoms
            # charges
            momenta=atoms.get_momenta(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
            constraint=atoms.constraints,
            calculator=atoms.calc,
            info=copy.deepcopy(atoms.info),
        )

        return scferr_atoms
    

if __name__ == "__main__":
    ...
