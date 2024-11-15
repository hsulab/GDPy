#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import List

import numpy as np

from ase import Atoms
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator


def read_cp2k_xyz(fpath):
    """Read xyz-like file by cp2k.

    Accept prefix-pos-1.xyz or prefix-frc-1.xyz.
    """
    # - read properties
    frame_steps, frame_times, frame_energies = [], [], []
    frame_symbols = []
    frame_properties = []  # coordinates or forces
    with open(fpath, "r") as fopen:
        while True:
            line = fopen.readline()
            if not line:
                break
            natoms = int(line.strip().split()[0])
            symbols, properties = [], []
            line = fopen.readline()  # energy line
            info_data = line.strip().split()
            frame_energies.append(info_data[-1])
            for i in range(natoms):
                line = fopen.readline()
                data_line = line.strip().split()
                symbols.append(data_line[0])
                properties.append(data_line[1:])
            frame_symbols.append(symbols)
            frame_properties.append(properties)

    return frame_symbols, frame_energies, frame_properties


INPUT_STRUCTURE_FLAG = ("MODULE", "QUICKSTEP:", "ATOMIC", "COORDINATES", "IN", "angstrom")

def check_input_structure_section(line):
    """Check the input structures from cp2k.out.

    v2022.1 has one more space before ATOMIC than v2024.1.

    """
    # data = tuple(line.strip().split())

    return line.strip().startswith("MODULE QUICKSTEP:")

def check_input_pbc_section(line):
    """"""
    return (
        line.strip().startswith("POISSON| Periodicity") or 
        line.strip().startswith("CELL_TOP| Periodicity")
    )


def read_cp2k_spc(wdir, prefix: str="cp2k"):
    """"""
    wdir = pathlib.Path(wdir)
    with open(wdir/f"{prefix}.out", "r") as fopen:
        lines = fopen.readlines()

    num_atoms = -1
    pbc, cell, structure = "", [], []
    energy, forces = None, []
    is_coord, is_force = False, False
    for line in lines:
        # cell
        if line.strip().startswith("CELL_TOP| Vector"):
            cell.append(line)
        if check_input_pbc_section(line):
            pbc = line.strip().split()[-1]
        # num of atoms
        if line.strip().startswith("- Atoms:"):
            if num_atoms < 0:
                num_atoms = int(line.strip().split()[-1])
            else:
                raise RuntimeError()
        # coordinates
        if check_input_structure_section(line):
            is_coord = True
            assert num_atoms > 0
        if is_coord:
            if len(structure) < num_atoms + 3:
                structure.append(line)
            else:
                is_coord = False
        # energy
        if line.strip().startswith("ENERGY| Total FORCE_EVAL"):
            energy = float(line.strip().split()[-1])
        # forces
        if line.strip().startswith("ATOMIC FORCES in [a.u.]"):
            is_force = True
        if line.strip().startswith("SUM OF ATOMIC FORCES"):
            is_force = False
        if is_force:
            forces.append(line)
    
    cell = np.array([c.strip().split()[4:7] for c in cell], dtype=np.float64)

    if pbc == "XYZ":
        pbc = True
    else:
        raise RuntimeError()

    coordinates = np.array([c.strip().split()[4:7] for c in structure[3:]], dtype=np.float64)
    symbols = [c.strip().split()[2] for c in structure[3:]]

    atoms = Atoms(symbols, positions=coordinates, cell=cell, pbc=pbc)  

    assert isinstance(energy, float)
    energy *= units.Hartree
    forces = np.array([frc.strip().split()[3:] for frc in forces[3:]], dtype=np.float64)
    forces *= units.Hartree/units.Bohr

    results = dict(
        energy=energy,
        free_energy=energy,
        forces=forces
    )

    calc = SinglePointCalculator(atoms, **results)
    atoms.calc = calc

    return atoms


def read_cp2k_energy_force(wdir, prefix: str="cp2k"):
    """"""
    wdir = pathlib.Path(wdir)
    with open(wdir/f"{prefix}.out", "r") as fopen:
        lines = fopen.readlines()

    "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:             `WHAT WE NEED`"
    " ATOMIC FORCES in [a.u.]"

    energy, forces = None, []
    is_force = False
    for line in lines:
        if line.strip().startswith("ENERGY| Total FORCE_EVAL"):
            energy = float(line.strip().split()[-1])
        if line.strip().startswith("ATOMIC FORCES in [a.u.]"):
            is_force = True
        if line.strip().startswith("SUM OF ATOMIC FORCES"):
            is_force = False
        if is_force:
            forces.append(line)

    assert isinstance(energy, float)
    energy *= units.Hartree
    forces = np.array([frc.strip().split()[3:] for frc in forces[3:]], dtype=np.float64)
    forces *= units.Hartree/units.Bohr

    results = dict(
        energy = energy,
        free_energy = energy,
        forces = forces
    )

    return results


def read_cp2k_outputs(wdir, prefix: str = "cp2k") -> List[Atoms]:
    """"""
    wdir = pathlib.Path(wdir)

    # positions
    pos_fpath = wdir / (prefix + "-pos-1.xyz")
    frame_symbols, frame_energies, frame_positions = read_cp2k_xyz(pos_fpath)
    # cp2k uses a.u. and we use eV
    frame_energies = np.array(frame_energies, dtype=np.float64)
    frame_energies *= units.Hartree  # 2.72113838565563E+01
    # cp2k uses AA the same as we do
    frame_positions = np.array(frame_positions, dtype=np.float64)

    # forces
    frc_fpath = wdir / (prefix + "-frc-1.xyz")
    _, _, frame_forces = read_cp2k_xyz(frc_fpath)
    # cp2k uses a.u. and we use eV/AA
    frame_forces = np.array(frame_forces, dtype=np.float64)
    frame_forces *= (
        units.Hartree / units.Bohr
    )  # (2.72113838565563E+01/5.29177208590000E-01)

    # - simulation box
    # parse cell from inp or out
    box_fpath = wdir / (prefix + "-1.cell")
    with open(box_fpath, "r") as fopen:
        # Step   Time [fs]
        # Ax [Angstrom]       Ay [Angstrom]       Az [Angstrom]
        # Bx [Angstrom]       By [Angstrom]       Bz [Angstrom]
        # Cx [Angstrom]       Cy [Angstrom]       Cz [Angstrom]      Volume [Angstrom^3]
        lines = fopen.readlines()
        data = np.array([line.strip().split() for line in lines[1:]], dtype=np.float64)
    steps = data[:, 0]
    boxes = data[:, 2:-1]

    # TODO: step must be int?
    # attach forces to frames, zip the shortest
    frames = []
    for step, symbols, box, positions, energy, forces in zip(
        steps, frame_symbols, boxes, frame_positions, frame_energies, frame_forces
    ):
        atoms = Atoms(
            symbols,
            positions=positions,
            cell=box.reshape(3, 3),
            pbc=[1, 1, 1],  # TODO: should determine in the cp2k input file
        )
        atoms.info["step"] = int(step)
        spc = SinglePointCalculator(
            atoms=atoms,
            energy=energy,
            free_energy=energy,  # TODO: depand on electronic method used
            forces=forces,
        )
        atoms.calc = spc
        frames.append(atoms)

    return frames


if __name__ == "__main__":
    ...
  
