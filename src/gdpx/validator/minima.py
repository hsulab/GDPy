#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from typing import Any, Optional

import numpy as np
import numpy.typing
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.io import read, write

from gdpx.builder.builder import StructureBuilder
from gdpx.data.array import AtomsNDArray
from gdpx.factory.builder import canonicalise_builder
from gdpx.validator.validator import BaseValidator
from gdpx.worker.drive import DriverBasedWorker

"""Validate minima and relative energies...
"""


def make_clean_atoms(atoms_, results=None):
    """Create a clean atoms from the input."""
    atoms = Atoms(
        symbols=atoms_.get_chemical_symbols(),
        positions=atoms_.get_positions().copy(),
        cell=atoms_.get_cell().copy(),
        pbc=copy.deepcopy(atoms_.get_pbc()),
    )
    if results is not None:
        spc = SinglePointCalculator(atoms, **results)
        atoms.calc = spc

    return atoms


def compare_structures(
    v_frames: list[Atoms],
    p_frames_ini: list[Atoms],
    p_frames_end: list[Atoms],
    energy_references: Optional[tuple[numpy.typing.NDArray, numpy.typing.NDArray]] = None,
):
    """"""
    # number of atoms
    v_natoms = np.array([len(a) for a in v_frames])
    p_natoms = np.array([len(a) for a in p_frames_end])
    assert np.allclose(v_natoms, p_natoms), "Number of atoms are not consistent."

    # total energies
    v_ene = np.array([a.get_potential_energy() for a in v_frames])
    p_ene_ini = np.array([a.get_potential_energy() for a in p_frames_ini])
    p_ene_end = np.array([a.get_potential_energy() for a in p_frames_end])

    ene_data = [v_ene, p_ene_ini, p_ene_end]

    if energy_references is not None:
        v_f_ene = v_ene - energy_references[0]  # validation formation energy
        p_f_ene_ini = p_ene_ini - energy_references[1]  # prediction formation energy at initial
        p_f_ene_end = p_ene_end - energy_references[1]  # prediction formation energy
        ene_data.extend([v_f_ene, p_f_ene_ini, p_f_ene_end])

    # maximum forces TODO: constraints?
    v_maxfrc = np.array([np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in v_frames])
    p_maxfrc_ini = np.array([np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in p_frames_ini])
    p_maxfrc_end = np.array([np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in p_frames_end])

    frc_data = [v_maxfrc, p_maxfrc_ini, p_maxfrc_end]

    # displacement
    disp = []  # displacements
    for ref_atoms, pre_atoms in zip(v_frames, p_frames_end):
        vector = pre_atoms.get_positions() - ref_atoms.get_positions()
        _, vlen = find_mic(vector, pre_atoms.get_cell())
        disp.append(vlen.max())

    results = dict(natoms=v_natoms, ene=ene_data, maxfrc=frc_data, disp=disp)

    return results


def summarise_validation(natoms, ene, maxfrc, disp, show_ranking: bool=False) -> str:
    """"""
    content = "# Name     N_a  " + ("{:>12s}  " * 9).format(
        "E_v", "E_p_ini", "E_p_end", "E_d_ini", "E_d_end", "Fmax_v", "Fmax_p_ini", "Fmax_p_end", "Disp"
    )
    line_format = "{:>6d}  " * 2 + "{:>12.4f}  " * 9

    num_ene_columns = len(ene)
    if num_ene_columns == 3:
        ...
    elif num_ene_columns == 6:
        content += ("{:>12s}  " * 3).format("Ef_v", "Ef_p_ini", "Ef_p_end")
        line_format += "{:>12.4f}  " * 3
    else:
        raise Exception(f"Unknown number of energy columns: {num_ene_columns}.")

    num_structures = len(natoms)

    if show_ranking:
        indices = np.arange(num_structures, dtype=np.int64)
        sort = np.argsort(ene[0])
        v_rankings = sorted(indices, key=lambda i: sort[i])
        sort = np.argsort(ene[1])
        p_rankings_ini = sorted(indices, key=lambda i: sort[i])
        sort = np.argsort(ene[2])
        p_rankings_end = sorted(indices, key=lambda i: sort[i])
        
        content += ("{:>6s}  " * 3).format("Erk_v", "Erk_p_ini", "Erk_p_end")
        line_format += "{:>6d}  " * 3

    content += "\n"
    line_format += "\n"

    for i in range(num_structures):
        ene_diff_ini = ene[1][i] - ene[0][i]
        ene_diff_end = ene[2][i] - ene[0][i]
        data = [
            ene[0][i],
            ene[1][i],
            ene[2][i],
            ene_diff_ini,
            ene_diff_end,
            maxfrc[0][i],
            maxfrc[1][i],
            maxfrc[2][i],
            disp[i],
        ]
        if num_ene_columns == 6:
            data.extend([ene[3][i], ene[4][i], ene[5][i]])  # formation energies
        if show_ranking:
            data.extend([v_rankings[i], p_rankings_ini[i], p_rankings_end[i]])
        content += line_format.format(i, natoms[i], *data)

    return content


def read_reference_structures(inp: list[Any]):
    """"""
    energy_list = []
    for x in inp:
        if isinstance(x, float):
            energies = [x]
        elif isinstance(x, dict):
            builder = canonicalise_builder(x)
            assert builder is not None
            structures = builder.run()
            energies = [a.get_potential_energy() for a in structures]
        else:
            raise Exception(f"Unknown {x} of type {type(x)}.")
        energy_list.append(energies)

    # Broadcast energies
    numbers = np.array([len(x) for x in energy_list])
    num_min = numbers.min()
    if np.all(numbers == num_min):
        reference_energies = np.array(energy_list).sum(axis=0)
    else:
        if num_min == 1 and np.sum(numbers != num_min) == 1:
            num_max = numbers.max()
            new_energy_list = []
            for energies in energy_list:
                if len(energies) == 1:
                    new_energy_list.append(energies * num_max)
                else:
                    new_energy_list.append(energies)
            reference_energies = np.array(new_energy_list).sum(axis=0)
        else:
            raise Exception(f"Number of energies are not consistent: {numbers}.")

    return reference_energies


class MinimaValidator(BaseValidator):
    """Run minimisation on various configurations and compare relative energy.

    TODO:

        Support the comparison of minimisation trajectories.

    """

    name: str = "minima"

    def __init__(self, formation_energy: Optional[dict] = None, show_ranking: bool = False, *args, **kwargs):
        """Initialise the validator.

        Args:
            show_ranking: Show the energetic ranking of the structures.

        """
        super().__init__(*args, **kwargs)

        self.show_ranking = show_ranking

        if formation_energy is not None:
            validation_energies = read_reference_structures(formation_energy["validation"])
            prediction_energies = read_reference_structures(formation_energy["prediction"])
            self.reference_energies = (validation_energies, prediction_energies)
        else:
            self.reference_energies = None

        return

    def run(self, structures: Optional[Any] = None, worker: Optional[DriverBasedWorker] = None, *args, **kwargs):
        """"""
        super().run()

        # Check what input structures we have
        if structures is not None:
            structure_sets = structures
            self._print("Use the structures at run time.")
        else:
            if isinstance(self.structures, (list, tuple)):
                structure_sets = self.structures
            else:
                # Assume it is a builder
                structure_sets = [self.structures, None]
            self._print("Use the structures at init time.")

        num_structure_sets = len(structure_sets)
        if num_structure_sets == 1:
            v_structures = structure_sets[0]
            p_structures = None
        elif num_structure_sets == 2:
            v_structures, p_structures = structure_sets
        else:
            raise Exception(f"{self.__class__.__name__} requires one or two sets of structures.")

        assert v_structures is not None, "Structures to validate must be provided either init or run time."

        if isinstance(v_structures, StructureBuilder):
            v_structures = v_structures.run()

        if isinstance(p_structures, StructureBuilder):
            p_structures = p_structures.run()

        # Make sure we have a worker to do the minimisations
        if p_structures is None:
            if worker is not None:
                v_worker = worker
                self._print("Use the worker at run time.")
            else:
                v_worker = self.worker
            assert v_worker is not None, "Worker must be provided either init or run time."
            v_worker.directory = self.directory / "_run"

            ini_frames, end_frames = self._irun(v_structures, v_worker)
        else:
            if isinstance(p_structures, AtomsNDArray):
                self._print(f"The minimised structures {p_structures=}.")
                if p_structures.ndim != 2:
                    raise Exception(f"Invalid prediction structures with ndim {p_structures.ndim}.")
                ini_frames, end_frames = [], []
                for traj in p_structures.tolist():
                    ini_frames.append(traj[0])
                    for atoms in traj[::-1]:
                        if atoms is not None:
                            end_frames.append(atoms)
                            break
            else:
                # Assume it is just a list of Atoms
                ini_frames, end_frames = p_structures, p_structures

        # Run the minimisation and compare the results
        is_finished = False

        if ini_frames is not None and end_frames is not None:
            results = compare_structures(
                v_structures, ini_frames, end_frames, energy_references=self.reference_energies
            )
            if not pathlib.Path(self.directory / "v.dat").exists():
                content = summarise_validation(**results, show_ranking=self.show_ranking)
                with open(self.directory / "v.dat", "w") as fopen:
                    fopen.write(content)
                is_finished = True
        else:
            is_finished = False

        return is_finished

    def _irun(
        self, frames: list[Atoms], worker: DriverBasedWorker
    ) -> tuple[Optional[list[Atoms]], Optional[list[Atoms]]]:
        """"""
        assert isinstance(worker, DriverBasedWorker), "Worker must be a DriverBasedWorker."

        cache_fpath = self.directory / "pred.xyz"
        if cache_fpath.exists():
            ini_frames = read(self.directory / "pred_ini.xyz", ":")
            end_frames = read(cache_fpath, ":")
            return ini_frames, end_frames

        _ = worker.run(frames)
        _ = worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            trajectories = worker.retrieve(include_retrieved=True)
            ini_frames = [t[0] for t in trajectories]
            write(self.directory / "pred_ini.xyz", ini_frames)
            end_frames = [t[-1] for t in trajectories]
            write(cache_fpath, end_frames)
        else:
            ini_frames, end_frames = None, None

        return ini_frames, end_frames  # type: ignore


if __name__ == "__main__":
    ...
