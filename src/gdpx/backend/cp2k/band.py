import pathlib

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator

from .parser import read_cp2k_xyz


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
                curr_xyzfile = wdir / f"cp2k-pos-Replica_nr_{i + 1}-1.xyz"
            else:
                curr_xyzfile = wdir / f"cp2k-pos-Replica_nr_{str(i + 1).zfill(2)}-1.xyz"
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
                    assert np.isclose(ene_in_pos, energies[i, j]), (
                        f"Energy in positions {ene_in_pos} does not match the output energy {energies[i, j]} for band {i} image {j}."
                    )
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
