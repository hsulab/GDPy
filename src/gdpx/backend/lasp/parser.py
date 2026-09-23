#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import io
import pathlib
import tarfile
import tempfile
import warnings
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.io import read, write


def is_number(s):
    """"""
    try:
        float(s)
        return True
    except ValueError:
        ...

    return False


def compare_trajectory_continuity(t0, t1):
    """Compare positions."""
    a0, a1 = t0[-1], t1[0]
    cell = a0.get_cell(complete=True)
    shift = a0.positions - a1.positions
    curr_vectors, curr_distances = find_mic(shift, cell, pbc=True)

    # Due to the floating point precision, arc -> atoms may lead to
    # position inconsistent after 8 decimals...
    return np.allclose(
        curr_vectors,
        np.zeros(curr_vectors.shape),
        # rtol=1e-05, atol=1e-08, equal_nan=False
        rtol=1e-04,
        atol=1e-06,
        equal_nan=False,
    )


def read_laspset(train_structures):
    """Read LASP TrainStr.txt and TrainFor.txt files."""
    train_structures = pathlib.Path(train_structures)
    frames = []

    all_energies, all_forces, all_stresses = [], [], []

    # Read TrainStr.txt
    # TODO: use yield
    with open(train_structures, "r") as fopen:
        while True:
            line = fopen.readline()
            if line.strip().startswith("Start one structure"):
                # - energy
                line = fopen.readline()
                energy = float(line.strip().split()[-2])
                all_energies.append(energy)
                # - natoms
                line = fopen.readline()
                natoms = int(line.strip().split()[-1])
                # skip 5 lines, symbol info and training weights
                # skipped_lines = [fopen.readline() for i in range(5)]
                # - cell
                cell = []
                for _ in range(1000):
                    lat_line = fopen.readline()
                    if lat_line.strip().startswith("lat"):
                        cell.append(lat_line.strip().split()[1:])
                    if len(cell) == 3:
                        break
                else:
                    raise RuntimeError("Failed to read lattice.")
                cell = np.array(cell, dtype=float)
                # - symbols, positions, and charges
                anumbers, positions, charges = [], [], []
                for i in range(natoms):
                    data = fopen.readline().strip().split()[1:]
                    anumbers.append(int(data[0]))
                    positions.append([float(x) for x in data[1:4]])
                    charges.append(float(data[-1]))
                atoms = Atoms(numbers=anumbers, positions=positions, cell=cell, pbc=True)
                assert fopen.readline().strip().startswith("End one structure")
                frames.append(atoms)
                # break
            if not line:
                break

    # Read TrainFor.txt
    train_forces = train_structures.parent / "TrainFor.txt"
    with open(train_forces, "r") as fopen:
        while True:
            line = fopen.readline()
            if line.strip().startswith("Start one structure"):
                # - stress, voigt order
                stress = np.array(fopen.readline().strip().split()[1:], dtype=float)
                # - symbols, forces
                anumbers, forces = [], []
                line = fopen.readline()
                while True:
                    if line.strip().startswith("force"):
                        data = line.strip().split()[1:]
                        anumbers.append(int(data[0]))
                        forces.append([float(x) for x in data[1:4]])
                    else:
                        all_forces.append(forces)
                        assert line.strip().startswith("End one structure")
                        break
                    line = fopen.readline()
                # break
            if not line:
                break

    for i, atoms in enumerate(frames):
        calc = SinglePointCalculator(atoms, energy=all_energies[i], forces=all_forces[i])
        atoms.calc = calc
    write(train_structures.parent / "dataset.xyz", frames)

    return frames


def read_lasp_structures(
    mdir: pathlib.Path, wdir: pathlib.Path, archive_path: Optional[pathlib.Path] = None
) -> list[Atoms]:
    """Read simulation trajectory in the dmol3 format.

    Note:
        The trajectory length may not equal to steps depending on the simulation tasks.
        The LBFGS minimisation has steps+2 frames.

    Args:
        mdir: Main directory path.
        wdir: Working directory path.
        archive_path: Archive path if any.

    Returns:
        A list of ASE Atoms objects.

    """
    # Check if output file exists...
    if (not (wdir / "allstr.arc").exists()) and archive_path is None:
        return []

    # Get IOs
    stru_io, afrc_io, lout_io = None, None, None
    if archive_path is None:
        with open(wdir / "allstr.arc", "r") as fopen:
            stru_io = io.StringIO(fopen.read())
        afrc_io = open(wdir / "allfor.arc", "r")  # atomic forces in arc format
        lout_io = open(wdir / "lasp.out", "r")
    else:
        rpath = wdir.relative_to(mdir.parent)
        stru_tarname = str(rpath / "allstr.arc")
        afrc_tarname = str(rpath / "allfor.arc")
        lout_tarname = str(rpath / "lasp.out")
        with tarfile.open(archive_path, "r:gz") as tar:
            for tarinfo in tar:
                if tarinfo.name.startswith(wdir.name):
                    if tarinfo.name == stru_tarname:
                        stru_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                    elif tarinfo.name == afrc_tarname:
                        afrc_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                    elif tarinfo.name == lout_tarname:
                        lout_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                    else:
                        ...
                else:
                    ...
            else:  # TODO: if not find target traj?
                ...

    # Three IOs must either io.StringIO or _io.TextIOWrapper
    assert stru_io is not None, f"Failed to get allstr.arc in {wdir.resolve()}."
    assert afrc_io is not None, f"Failed to get allfor.arc in {wdir.resolve()}."
    assert lout_io is not None, f"Failed to get lasp.out   in {wdir.resolve()}."

    # Check extra files
    vel_io = None
    if archive_path is None:
        if (wdir / "vel.arc").exists():
            with open(wdir / "vel.arc", "r") as fopen:
                vel_io = io.StringIO(fopen.read())
    else:
        rpath = wdir.relative_to(mdir.parent)
        vel_tarname = str(rpath / "vel.arc")
        with tarfile.open(archive_path, "r:gz") as tar:
            for tarinfo in tar:
                if tarinfo.name == vel_tarname:
                    vel_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                else:
                    ...
            else:
                ...

    # Parse arc structures, and ASE does not support read dmol-arc from stringIO
    with tempfile.NamedTemporaryFile(mode="w", suffix=".arc") as tmp:
        tmp.write(stru_io.getvalue())
        tmp.seek(0)
        traj_frames = read(tmp.name, ":", format="dmol-arc")
    natoms = len(traj_frames[-1])

    # Read energy, forces, stress from allfor.arc
    traj_steps = []
    traj_energies = []
    traj_stress = []
    traj_forces = []
    while True:
        line = afrc_io.readline()
        if line.strip().startswith("For"):
            step = int(line.split()[1])
            traj_steps.append(step)
            # Check if energy is a number, some ill structure may result in large energy of ******
            energy_data = line.split()[3]
            try:
                energy = float(energy_data)
            except ValueError:
                energy = np.inf
                msg = "Energy is too large at {}. The structure maybe ill-constructed.".format(wdir)
                warnings.warn(msg, UserWarning)
            traj_energies.append(energy)
            # stress
            line = afrc_io.readline()
            stress = np.array(line.split())  # eV/Ang^3
            assert stress.shape[0] == 6
            traj_stress.append(stress)
            # forces
            forces = []
            for _ in range(natoms):
                line = afrc_io.readline()
                force_data = line.strip().split()
                if len(force_data) == 3:  # expect three numbers
                    force_data_ = []
                    for x in force_data:
                        if not is_number(x):
                            force_data_.append(np.inf)
                        else:
                            force_data_.append(float(x))
                    force_data = force_data_
                else:  # too large forces make out become ******
                    force_data = [np.inf] * 3
                forces.append(force_data)
            forces = np.array(forces, dtype=float)
            traj_forces.append(forces)
        if line.strip() == "":
            ...
        if not line:  # if line == "":
            break
    assert len(traj_frames) == len(traj_steps), f"Output number is inconsistent in {wdir.resolve()}."

    # Read velocities if any
    traj_velocities = []
    if vel_io is not None:
        while True:
            line = vel_io.readline()
            if "Time" in line:
                data = line.strip().split()
                time = float(data[3])  # in fs
                timestep = float(data[7])  # in fs
                step = int(time / timestep)
                assert step in traj_steps, f"Step {step} in allvel.arc not in allstr.arc in {wdir.resolve()}."
                # velocities
                velocities = []
                for _ in range(natoms):
                    line = vel_io.readline()
                    velocity_data = line.strip().split()[1:]
                    if len(velocity_data) == 3:  # expect three numbers
                        velocity_data_ = []
                        for x in velocity_data:
                            if not is_number(x):
                                velocity_data_.append(np.inf)
                            else:
                                velocity_data_.append(float(x))
                        velocity_data = velocity_data_
                    else:  # too large velocities make out become ******
                        velocity_data = [np.inf] * 3
                    velocities.append(velocity_data)
                velocities = np.array(velocities, dtype=float)
                traj_velocities.append(velocities)
            if line.strip() == "":
                ...
            if not line:  # if line == "":
                break
        assert len(traj_velocities) == len(traj_steps), f"Velocity number is inconsistent in {wdir.resolve()}."
    else:
        ...

    # Create the trajectory with spc
    for i, atoms in enumerate(traj_frames):
        calc = SinglePointCalculator(
            atoms,
            energy=traj_energies[i],
            forces=traj_forces[i],
            stress=traj_stress[i],
        )
        atoms.calc = calc

    if traj_velocities:
        for i, atoms in enumerate(traj_frames):
            atoms.set_velocities(traj_velocities[i])

    # Check if the structure is too bad.
    TOO_SHORT_BOND_TAG = "Warning: Minimum Structure with too short bond"  # v3.3.4
    is_badstru = False
    lines = lout_io.readlines()
    for line in lines:
        if TOO_SHORT_BOND_TAG in line:
            is_badstru = True
            break
        else:
            ...
    traj_frames[-1].info["is_badstru"] = is_badstru

    # Close IOs
    stru_io.close()
    afrc_io.close()
    lout_io.close()
    if vel_io is not None:
        vel_io.close()

    return traj_frames


if __name__ == "__main__":
    ...
