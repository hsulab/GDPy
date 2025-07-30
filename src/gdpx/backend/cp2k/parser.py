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
    frame_steps = []
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
            step = int(info_data[2][:-1])  # step number, remove the comma
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
                frame_steps.append(step)
                frame_energies.append(energy)
                frame_symbols.append(symbols)
                frame_properties.append(properties)

    return frame_steps, frame_symbols, frame_energies, frame_properties


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
    """Read the cp2k outputs from a calculation with run_type of `GEO_OPT`, `CELL_OPT` or `MD`.

    The GEO_OPT does not write positions and forces for the input structure to xyz files, thus, the trajectory starts
    with the first frame of the optimisation.

    """
    # Read positions
    pos_fpath = wdir / (prefix + "-pos-1.xyz")
    frame_steps, frame_symbols, frame_energies, frame_positions = read_cp2k_xyz(pos_fpath)
    # cp2k uses a.u. and we use eV
    frame_energies = np.array(frame_energies, dtype=np.float64)
    frame_energies *= units.Hartree  # 2.72113838565563E+01
    # cp2k uses AA the same as we do
    frame_positions = np.array(frame_positions, dtype=np.float64)
    num_frames_by_pos = frame_positions.shape[0]

    # Read forces
    frc_fpath = wdir / (prefix + "-frc-1.xyz")
    _, _, _, frame_forces = read_cp2k_xyz(frc_fpath)
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
    steps = np.array(data[:, 0], dtype=np.int64)  # step numbers
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
    for i, (step, symbols, box, positions, energy, forces) in enumerate(
        zip(steps, frame_symbols, boxes, frame_positions, frame_energies, frame_forces)
    ):
        atoms = Atoms(
            symbols,
            positions=positions,
            cell=box.reshape(3, 3),
            pbc=[1, 1, 1],  # TODO: should determine in the cp2k input file
        )
        atoms.info["step"] = int(step)
        assert step == frame_steps[i], f"Step {step} does not match the expected step {frame_steps[i]}."
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


def read_cp2k_output_from_band(
    wdir: pathlib.Path, prefix: str = "cp2k", print_func=print, debug_func=print
) -> list[list[Atoms]]:
    """Read the cp2k output from a calculation with `RUN_TYPE BAND`.

    CP2K does not save forces to the xyz files, thus, we read forces from the output file.

    """
    print_func(f"***** read_trajectory at {str(wdir)} *****")
    cell = None  # TODO: if no pbc?
    num_atoms = None
    num_images = None
    temp_forces, temp_energies = [], []
    energies, forces = [], []
    with open(wdir / f"{prefix}.out", "r") as fopen:
        while True:
            line = fopen.readline()
            if not line:
                break
            # read cell
            if "CELL| Volume" in line:
                found_cell = False
                cell_data = []
                for i in range(3):
                    line = fopen.readline()
                    if line:
                        cell_data.append(line)
                    else:
                        break
                else:
                    found_cell = True
                if found_cell:
                    try:
                        cell = [x.strip().split()[4:7] for x in cell_data]
                    except Exception:
                        debug_func("cell is not found.")
                        break
            # read natoms
            if "TOTAL NUMBERS AND MAXIMUM NUMBERS" in line:
                found_natoms = False
                for i in range(3):
                    line = fopen.readline()
                    if not line:
                        break
                else:
                    found_natoms = True
                if found_natoms:
                    try:
                        num_atoms = int(line.strip().split()[-1])
                        debug_func(f"{num_atoms=}")
                    except Exception:
                        debug_func("num_atoms is not found.")
                        break
                else:
                    break
            if "Number of Images" in line:
                # line = fopen.readline() # BUG: inconsistent Images and Replicas?
                if not line:
                    break
                try:
                    num_images = int(line.strip().split()[-2])
                    debug_func(line)
                    debug_func(f"{num_images=}")
                except Exception:
                    debug_func("num_images is not found.")
            # NOTE: For method with LineSearch, several SCF may be performed at one step
            if "Computing Energies and Forces" in line:
                assert num_atoms is not None, "num_atoms is not read properly."
                assert num_images is not None, "num_images is not read properly."
                # NEB| REPLICA Nr.    1- Energy and Forces
                # NEB|                                     Total energy:       -2940.286865478840
                # NEB|    ATOM                            X                Y                Z
                curr_data = []
                found_replica_forces = False
                for i in range((num_atoms + 3) * num_images):
                    line = fopen.readline()
                    if line:
                        curr_data.append(line)
                    else:
                        break
                else:
                    # current replica's forces are complete...
                    found_replica_forces = True
                if found_replica_forces:
                    curr_energies = [
                        float(curr_data[i].strip().split()[-1]) for i in range(1, len(curr_data), num_atoms + 3)
                    ]
                    temp_energies.append(curr_energies)
                    curr_forces = []
                    for ir in range(num_images):
                        curr_forces.append(
                            [
                                curr_data[i].strip().split()[2:]
                                for i in range(
                                    (num_atoms + 3) * ir + 3,
                                    (num_atoms + 3) * ir + 3 + num_atoms,
                                )
                            ]
                        )
                    temp_forces.append(curr_forces)
                else:
                    break
            if "BAND TOTAL ENERGY" in line:
                if temp_energies and temp_forces:  # if the step completed...
                    energies.append(temp_energies[-1])
                    forces.extend(temp_forces[-1])
                    temp_forces, temp_energies = [], []

    # Assemble the results into atoms objects
    assert num_atoms is not None, "num_atoms is not read properly."
    assert num_images is not None, "num_images is not read properly."

    frames = []  # shape (nbands, nimages)
    if forces:
        # truncate to the last complete band as output in cp2k.out
        forces = np.array(forces, dtype=np.float64)
        shape = forces.shape
        debug_func(f"forces: {shape}")
        num_bands_in_out = int(shape[0] / num_images)
        forces = forces[: num_bands_in_out * num_images]
        debug_func(f"truncated forces: {forces.shape} nbands: {num_bands_in_out}")
        forces = np.reshape(
            forces, (num_bands_in_out, num_images, num_atoms, -1)
        )  # shape (nbands, nimages, natoms, 3)
        forces *= units.Hartree / units.Bohr

        energies = np.array(energies)[: num_bands_in_out * num_images].reshape(num_bands_in_out, num_images)
        energies *= units.Hartree
        debug_func(f"energies: {energies.shape} nbands: {num_bands_in_out}")

        cell = np.array(cell, dtype=np.float64)
        debug_func(f"cell: {cell}")

        # read positions
        band_frames = []  # shape (nimages, nbands)
        for i in range(num_images):
            if num_images < 10:
                curr_xyzfile = wdir / f"cp2k-pos-Replica_nr_{i+1}-1.xyz"
            else:
                curr_xyzfile = wdir / f"cp2k-pos-Replica_nr_{str(i+1).zfill(2)}-1.xyz"
            # curr_frames = read(curr_xyzfile, index=":", format="xyz")[:num_bands]
            frame_steps, frame_symbols, frame_energies, frame_positions = read_cp2k_xyz(curr_xyzfile)
            curr_frames = []
            for _, symbols, positions, energy in zip(frame_steps, frame_symbols, frame_positions, frame_energies):
                if num_atoms != len(symbols):
                    raise Exception(
                        f"Number of atoms {num_atoms} does not match the number of symbols {len(symbols)} in {curr_xyzfile}."
                    )
                atoms = Atoms(symbols, positions=np.array(positions, dtype=np.float64), cell=cell, pbc=True)
                spc = SinglePointCalculator(atoms, energy=energy * units.Hartree, free_energy=energy * units.Hartree)
                atoms.calc = spc
                curr_frames.append(atoms)
            band_frames.append(curr_frames)

        # check if all replicas have the same number of frames
        num_bands_in_replicas = [len(band_frames[i]) for i in range(num_images)]
        num_bands_in_pos = min(num_bands_in_replicas)
        print(f"{num_bands_in_out=}  {num_bands_in_pos=}")
        num_bands = num_bands_in_pos
        if all([nb == num_bands_in_pos for nb in num_bands_in_replicas]):
            ...
        else:
            print_func(
                f"Number of bands in replicas is not the same: {num_bands_in_replicas}. "
                f"Truncating to the minimum number of bands {num_bands}."
            )
            for i in range(num_images):
                band_frames[i] = band_frames[i][:num_bands]

        # reshape frames_ to (num_bands, num_images)
        for j in range(num_bands):
            curr_band = []
            for i in range(num_images):
                curr_band.append(band_frames[i][j])
            frames.append(curr_band)

        # update calc results
        for i in range(num_bands):
            for j in range(num_images):
                atoms = frames[i][j]
                atoms.set_cell(cell)
                atoms.pbc = True
                if i < num_bands_in_out:
                    ene_in_pos = atoms.get_potential_energy()
                    assert np.isclose(
                        ene_in_pos, energies[i, j]
                    ), f"Energy in positions {ene_in_pos} does not match the output energy {energies[i, j]} for band {i} image {j}."
                    spc = SinglePointCalculator(
                        atoms,
                        energy=energies[i, j],
                        free_energy=energies[i, j],
                        forces=forces[i, j].copy(),
                    )
                    atoms.calc = spc
                else:
                    ...  # no forces available in the output.
    else:
        ...  # no forces available indicating that not even on band step is finished.

    return frames


#: Test on cp2k:v2022.1
UNCONVERGED_SCF_FLAG: str = "*** WARNING in qs_scf.F:598 :: SCF run NOT converged ***"

#: Test on cp2k:v2022.1
ABORT_FLAG: str = "ABORT"


def read_cp2k_scf_convergence(out_fpath: pathlib.Path) -> bool:
    """Read SCF convergence.

    Check if there is any unconverged SCF flag in the output file.

    """
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


def read_cp2k_program_convergence(out_fpath: pathlib.Path) -> bool:
    """Read the end of the cp2k program.

    Check the end of the cp2k output file to see if the program ended successfully.

    """
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
