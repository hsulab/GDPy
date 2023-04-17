#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pathlib

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

"""Convert cp2k md outputs to ase xyz file.
"""

def read_cp2k_xyz(fpath):
    """Read xyz-like file by cp2k.

    Accept prefix-pos-1.xyz or prefix-frc-1.xyz.
    """
    # - read properties
    frame_steps, frame_times, frame_energies = [], [], []
    frame_symbols = []
    frame_properties = [] # coordinates or forces
    with open(fpath, "r") as fopen:
        while True:
            line = fopen.readline()
            if not line:
                break
            natoms = int(line.strip().split()[0])
            symbols, properties = [], []
            line = fopen.readline() # energy line
            frame_energies.append(line.strip().split()[-1])
            for i in range(natoms):
                line = fopen.readline()
                data_line = line.strip().split()
                symbols.append(data_line[0])
                properties.append(data_line[1:])
            frame_symbols.append(symbols)
            frame_properties.append(properties)

    return frame_symbols, frame_energies, frame_properties


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("PREFIX", help="path with prefix such as ../prefix")
    args = parser.parse_args()

    p = pathlib.Path(args.PREFIX)
    wdir, prefix = p.parent, p.name
    pos_fpath = wdir / (prefix+"-pos-1.xyz")
    frc_fpath = wdir / (prefix+"-frc-1.xyz")

    # - positions
    frame_symbols, frame_energies, frame_positions = read_cp2k_xyz(pos_fpath)
    # NOTE: cp2k uses a.u. and we use eV
    frame_energies = np.array(frame_energies, dtype=np.float64)
    frame_energies *= 2.72113838565563E+01
    # NOTE: cp2k uses AA the same as we do
    frame_positions = np.array(frame_positions, dtype=np.float64)
    #frame_positions *= 5.29177208590000E-01
    print("shape of positions: ", frame_positions.shape)

    # - forces
    _, _, frame_forces = read_cp2k_xyz(frc_fpath)
    # NOTE: cp2k uses a.u. and we use eV/AA
    frame_forces = np.array(frame_forces, dtype=np.float64)
    frame_forces *= (2.72113838565563E+01/5.29177208590000E-01)
    print("shape of forces: ", frame_forces.shape)

    # - simulation box
    # TODO: parse cell from inp or out
    #cell = 12.490*np.eye(3)
    cell = 38.*np.eye(3)
    cell[2,2] = 12.

    # attach forces to frames
    frames = []
    for symbols, positions, energy, forces in zip(frame_symbols, frame_positions, frame_energies, frame_forces):
        atoms = Atoms(
                symbols, positions=positions,
                cell=cell, pbc=[1,1,1]
        )
        spc = SinglePointCalculator(atoms=atoms, energy=energy, forces=forces)
        atoms.calc = spc
        frames.append(atoms)
    write(wdir/(prefix+"-MDtraj.xyz"), frames)
    