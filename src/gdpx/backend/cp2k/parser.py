#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator


def read_cp2k_xyz(fpath: pathlib.Path):
    """Read xyz-like file by cp2k.

    Accept prefix-pos-1.xyz or prefix-frc-1.xyz.

    """
    # Read properties
    frame_energies = []
    frame_symbols = []
    frame_properties = []  # coordinates or forces
    with open(fpath, "r") as fopen:
        while True:
            # read the first line with the number of atoms
            line = fopen.readline()
            if not line:
                break
            num_atoms = int(line.strip().split()[0])
            # read the second line with step `i` and energy `E`
            line = fopen.readline()  # energy line
            if not line:
                break
            info_data = line.strip().split()
            energy = float(info_data[-1])  # energy in a.u.
            # read the next `num_atoms` lines with symbols and properties
            symbols, properties = [], []
            for _ in range(num_atoms):
                line = fopen.readline()
                if not line:
                    break
                data_line = line.strip().split()
                symbols.append(data_line[0])
                properties.append(data_line[1:])
            else:
                # only when all `num_atoms` lines are read then the results are appended
                frame_energies.append(energy)
                frame_symbols.append(symbols)
                frame_properties.append(properties)

    return frame_symbols, frame_energies, frame_properties


INPUT_STRUCTURE_FLAG = (
    "MODULE",
    "QUICKSTEP:",
    "ATOMIC",
    "COORDINATES",
    "IN",
    "angstrom",
)


def check_input_structure_section(line):
    """Check the input structures from cp2k.out.

    v2022.1 has one more space before ATOMIC than v2024.1.

    """
    # data = tuple(line.strip().split())

    return line.strip().startswith("MODULE QUICKSTEP:")


def check_input_pbc_section(line):
    """"""
    return line.strip().startswith("POISSON| Periodicity") or line.strip().startswith("CELL_TOP| Periodicity")


def read_cp2k_output_from_energy_force(wdir: pathlib.Path, prefix: str = "cp2k") -> Optional[Atoms]:
    """Read the cp2k output from a calculation with `RUN_TYPE ENERGY_FORCE`.

    This function is tested on CP2K v2022.1 but should be able to work across various versions.
    It tries to read the cell, pbc, natoms, positions, energy, and forces from the output.
    Sometimes, the calculation stops in the middle of the SCF and no energy and forces are written,
    thus, we return None that can be checked in read_trajectory and make read_convergence gives false.

    The `wdir/f"{prefix}.out"` must exist and not be empty.

    Args:
        wdir: The cp2k calculation directory.
        prefix: The name of the output file.

    Returns:
        An atoms object if the single-point calculation finished successfully, otherwise, None.

    """
    cp2k_out_fpath = wdir / f"{prefix}.out"
    with open(cp2k_out_fpath, "r") as fopen:
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
                raise RuntimeError(f"Cannot read `- Atoms:` at {str(wdir)}.")
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
        # The forces section is as follows:
        # ATOMIC FORCES in [a.u.]
        #
        # # Atom   Kind   Element          X              Y              Z
        #      1      1      C          -0.02038329    -0.02023729     0.01704284
        #      2      1      C           0.04225810    -0.00166965    -0.01513970
        #      3      2      O          -0.00234658    -0.04231827     0.00729175
        if line.strip().startswith("ATOMIC FORCES in [a.u.]"):
            is_force = True
        if line.strip().startswith("SUM OF ATOMIC FORCES"):
            is_force = False
        if is_force:
            forces.append(line)

    # The calculation did not event start if num_atoms is not properly read.
    if num_atoms < 0:
        return None

    cell = np.array([c.strip().split()[4:7] for c in cell], dtype=np.float64)

    if pbc == "XYZ":
        pbc = True
    else:
        raise RuntimeError(f"Cannot read a calculation with pbc != XYZ.")

    coordinates = np.array([c.strip().split()[4:7] for c in structure[3:]], dtype=np.float64)
    symbols = [c.strip().split()[2] for c in structure[3:]]

    atoms = Atoms(symbols, positions=coordinates, cell=cell, pbc=pbc)

    # assert isinstance(energy, float), f"Cannot convert energy `{energy}` to a float at `{str(wdir)}`."
    if energy is not None and len(forces) == num_atoms + 3:
        energy *= units.Hartree
        forces = np.array([frc.strip().split()[3:] for frc in forces[3:]], dtype=np.float64)
        forces *= units.Hartree / units.Bohr

        results = dict(energy=energy, free_energy=energy, forces=forces)

        calc = SinglePointCalculator(atoms, **results)
        atoms.calc = calc
    else:
        atoms = None

    return atoms


def read_cp2k_energy_force(wdir, prefix: str = "cp2k"):
    """"""
    wdir = pathlib.Path(wdir)
    with open(wdir / f"{prefix}.out", "r") as fopen:
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
    forces *= units.Hartree / units.Bohr

    results = dict(energy=energy, free_energy=energy, forces=forces)

    return results


def read_cp2k_outputs(wdir: pathlib.Path, prefix: str = "cp2k", task: str = "min") -> list[Atoms]:
    """Read the cp2k outputs from a calculation with run_type of `GEO_OPT`, `CELL_OPT` or `MD`."""
    # Read positions
    pos_fpath = wdir / (prefix + "-pos-1.xyz")
    frame_symbols, frame_energies, frame_positions = read_cp2k_xyz(pos_fpath)
    # cp2k uses a.u. and we use eV
    frame_energies = np.array(frame_energies, dtype=np.float64)
    frame_energies *= units.Hartree  # 2.72113838565563E+01
    # cp2k uses AA the same as we do
    frame_positions = np.array(frame_positions, dtype=np.float64)
    num_frames_by_pos = frame_positions.shape[0]

    # Read forces
    frc_fpath = wdir / (prefix + "-frc-1.xyz")
    _, _, frame_forces = read_cp2k_xyz(frc_fpath)
    # cp2k uses a.u. and we use eV/AA
    frame_forces = np.array(frame_forces, dtype=np.float64)
    frame_forces *= units.Hartree / units.Bohr  # (2.72113838565563E+01/5.29177208590000E-01)
    num_frames_by_frc = frame_forces.shape[0]

    # Read cells
    box_fpath = wdir / (prefix + "-1.cell")
    with open(box_fpath, "r") as fopen:
        # Each row contains:
        # """
        # Step   Time [fs]
        # Ax [Angstrom]       Ay [Angstrom]       Az [Angstrom]
        # Bx [Angstrom]       By [Angstrom]       Bz [Angstrom]
        # Cx [Angstrom]       Cy [Angstrom]       Cz [Angstrom]      Volume [Angstrom^3]
        # """
        lines = fopen.readlines()
        data = np.array([line.strip().split() for line in lines[1:]], dtype=np.float64)
    steps = data[:, 0]
    boxes = data[:, 2:-1]
    num_frames_by_box = boxes.shape[0]

    # Check consistency in num_frames
    num_frames = min((num_frames_by_pos, num_frames_by_frc, num_frames_by_box))
    if task == "min" or task == "cmin":
        # The cp2k optimisation will reevaluate the structure when converged,
        # thus, positions and forces are written for the reevaluated one but cell is not,
        # which means we have equal or greater number of frames in positions and forces.
        # TODO: Replace the last frame with the reevaluated one?
        if not (num_frames >= num_frames_by_box):
            raise Exception(
                f"Inconsistent number of frames in positions ({num_frames_by_pos}), forces ({num_frames_by_frc}), and boxes ({num_frames_by_box})."
            )
        else:
            ...
    else:  # md
        ...

    # Take the shortest number of frames by zip
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
    num_frames_ret = len(frames)  # returned num of frames
    assert (
        num_frames_ret == num_frames
    ), f"Inconsistent number of frames returned ({num_frames_ret}) and expected ({num_frames})."

    return frames


#: Test on cp2k:v2022.1
UNCONVERGED_SCF_FLAG: str = "*** WARNING in qs_scf.F:598 :: SCF run NOT converged ***"

#: Test on cp2k:v2022.1
ABORT_FLAG: str = "ABORT"


def read_cp2k_convergence(out_fpath: pathlib.Path) -> bool:
    """Read SCF convergence."""
    cp2kout = out_fpath

    converged = True
    with open(cp2kout, "r") as fopen:
        while True:
            line = fopen.readline()
            if not line:
                break
            if line.strip() == UNCONVERGED_SCF_FLAG:
                converged = False
                break
            if ABORT_FLAG in line:
                converged = False
                break

    return converged


CP2K_PROGRAM_END_FLAG: str = "PROGRAM ENDED AT"


def read_cp2k_spc_convergence(out_fpath: pathlib.Path) -> bool:
    """"""
    converged = False
    with open(out_fpath, "r") as fopen:
        while True:
            line = fopen.readline()
            if not line:
                break
            if CP2K_PROGRAM_END_FLAG in line:
                converged = True
                break

    return converged


if __name__ == "__main__":
    ...
