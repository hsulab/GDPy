#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Optional, TextIO, Tuple

import numpy as np

from ase import Atoms
from ase import units
from ase.io.vasp import _symbol_count_string, _symbol_count_from_symbols, _handle_ase_constraints
from ase.utils import writer


@writer
def write_vasp(
    fd: TextIO,
    atoms: Atoms,
    direct: bool = False,
    sort: bool = False,
    symbol_count: Optional[List[Tuple[str, int]]] = None,
    vasp5: bool = True,
    vasp6: bool = False,
    ignore_constraints: bool = False,
    write_velocities: bool = False,
    potential_mapping: Optional[dict] = None,
) -> None:
    """Method to write VASP position (POSCAR/CONTCAR) files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. Cartesian coordinates is default and default label is the
    atomic species, e.g. 'C N H Cu'.

    Args:
        fd (TextIO): writeable Python file descriptor
        atoms (ase.Atoms): Atoms to write
        direct (bool): Write scaled coordinates instead of cartesian
        sort (bool): Sort the atomic indices alphabetically by element
        symbol_count (list of tuples of str and int, optional): Use the given
            combination of symbols and counts instead of automatically compute
            them
        vasp5 (bool): Write to the VASP 5+ format, where the symbols are
            written to file
        vasp6 (bool): Write symbols in VASP 6 format (which allows for
            potential type and hash)
        ignore_constraints (bool): Ignore all constraints on `atoms`
        potential_mapping (dict, optional): Map of symbols to potential file
            (and hash). Only works if `vasp6=True`. See `_symbol_string_count`

    Raises:
        RuntimeError: raised if any of these are true:

            1. `atoms` is not a single `ase.Atoms` object.
            2. The cell dimensionality is lower than 3 (0D-2D)
            3. One FixedPlane normal is not parallel to a unit cell vector
            4. One FixedLine direction is not parallel to a unit cell vector
    """
    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError(
                "Only one atomic structure can be saved to VASP "
                "POSCAR/CONTCAR. Several were given."
            )
        else:
            atoms = atoms[0]

    # Check lattice vectors are finite
    if atoms.cell.rank < 3:
        raise RuntimeError(
            "Lattice vectors must be finite and non-parallel. At least "
            "one lattice length or angle is zero."
        )

    # Write atomic positions in scaled or cartesian coordinates
    if direct:
        coord = atoms.get_scaled_positions(wrap=False)
    else:
        coord = atoms.positions

    # Convert ASE constraints to VASP POSCAR constraints
    constraints_present = atoms.constraints and not ignore_constraints
    if constraints_present:
        sflags = _handle_ase_constraints(atoms)

    # Conditionally sort ordering of `atoms` alphabetically by symbols
    if sort:
        ind = np.argsort(atoms.symbols)
        symbols = atoms.symbols[ind]
        coord = coord[ind]
        if constraints_present:
            sflags = sflags[ind]
        if write_velocities:
            velocities = atoms.get_velocities()
            velocities = velocities[ind]*units.fs
    else:
        symbols = atoms.symbols
        if write_velocities:
            velocities = atoms.get_velocities()
            velocities *= units.fs

    # Set or create a list of (symbol, count) pairs
    sc = symbol_count or _symbol_count_from_symbols(symbols)

    # Write header as atomic species in `symbol_count` order
    label = " ".join(f"{sym:2s}" for sym, _ in sc)
    fd.write(label + "\n")

    # For simplicity, we write the unitcell in real coordinates, so the
    # scaling factor is always set to 1.0.
    fd.write(f"{1.0:19.16f}\n")

    for vec in atoms.cell:
        fd.write("  " + " ".join([f"{el:21.16f}" for el in vec]) + "\n")

    # Write version-dependent species-and-count section
    sc_str = _symbol_count_string(sc, vasp5, vasp6, potential_mapping)
    fd.write(sc_str)

    # Write POSCAR switches
    if constraints_present:
        fd.write("Selective dynamics\n")

    fd.write("Direct\n" if direct else "Cartesian\n")

    # Write atomic positions and, if any, the cartesian constraints
    for iatom, atom in enumerate(coord):
        for dcoord in atom:
            fd.write(f" {dcoord:19.16f}")
        if constraints_present:
            flags = ["F" if flag else "T" for flag in sflags[iatom]]
            fd.write("".join([f"{f:>4s}" for f in flags]))
        fd.write("\n")

    # Write velocties
    if write_velocities:
        fd.write("\n")
        for iatom, atom in enumerate(velocities):
            for dvelocity in atom:
                fd.write(f" {dvelocity:>15.8E}")
            fd.write("\n")
        fd.write("\n")

    return


if __name__ == "__main__":
    ...
