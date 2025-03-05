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
    p_frames: list[Atoms],
    energy_references: Optional[tuple[numpy.typing.NDArray, numpy.typing.NDArray]] = None,
):
    """"""
    # number of atoms
    v_natoms = np.array([len(a) for a in v_frames])
    p_natoms = np.array([len(a) for a in p_frames])
    assert np.allclose(v_natoms, p_natoms), "Number of atoms are not consistent."

    # total energies
    v_ene = np.array([a.get_potential_energy() for a in v_frames])
    p_ene = np.array([a.get_potential_energy() for a in p_frames])

    if energy_references is None:
        ene_data = (v_ene, p_ene)
    else:
        v_f_ene = v_ene - energy_references[0]  # validation formation energy
        p_f_ene = p_ene - energy_references[1]  # prediction formation energy
        ene_data = (v_ene, p_ene, v_f_ene, p_f_ene)

    # maximum forces TODO: constraints?
    v_maxfrc = np.array([np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in v_frames])
    p_maxfrc = np.array([np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in p_frames])

    # displacement
    disp = []  # displacements
    for ref_atoms, pre_atoms in zip(v_frames, p_frames):
        vector = pre_atoms.get_positions() - ref_atoms.get_positions()
        _, vlen = find_mic(vector, pre_atoms.get_cell())
        disp.append(np.linalg.norm(vlen))

    results = dict(natoms=v_natoms, ene=ene_data, maxfrc=(v_maxfrc, p_maxfrc), disp=disp)

    return results


def summarise_validation(natoms, ene, maxfrc, disp) -> str:
    """"""
    num_ene_columns = len(ene)

    if num_ene_columns == 2:
        line_format = "{:>6d}  " * 2 + "{:>12.4f}  " * 7 + "\n"

        content = (
            "# Name     N_a  "
            + ("{:>12s}  " * 7).format("E_v", "E_p", "E_d", "E_d/N_a", "Fmax_v", "Fmax_p", "Disp")
            + "\n"
        )

        num_structures = len(natoms)
        for i in range(num_structures):
            ene_diff = ene[0][i] - ene[1][i]
            data = [ene[0][i], ene[1][i], ene_diff, ene_diff / natoms[i], maxfrc[0][i], maxfrc[1][i], disp[i]]
            content += line_format.format(i, natoms[i], *data)
    elif num_ene_columns == 4:
        line_format = "{:>6d}  " * 2 + "{:>12.4f}  " * 9 + "\n"

        content = (
            "# Name     N_a  "
            + ("{:>12s}  " * 9).format("E_v", "E_p", "E_d", "E_d/N_a", "Ef_v", "Ef_p", "Fmax_v", "Fmax_p", "Disp")
            + "\n"
        )

        num_structures = len(natoms)
        for i in range(num_structures):
            ene_diff = ene[0][i] - ene[1][i]
            data = [
                ene[0][i],
                ene[1][i],
                ene_diff,
                ene_diff / natoms[i],
                ene[2][i],
                ene[3][i],
                maxfrc[0][i],
                maxfrc[1][i],
                disp[i],
            ]
            content += line_format.format(i, natoms[i], *data)
    else:
        raise Exception(f"Unknown number of energy columns: {num_ene_columns}.")

    return content


def summarise_validation_with_ranking(natoms, ene, maxfrc, disp) -> str:
    """"""
    line_format = "{:>6d}  " * 2 + "{:>12.4f}  " * 7 + "{:>6d}  " * 2 + "\n"

    content = (
        "# Name     N_a  "
        + ("{:>12s}  " * 7).format("E_v", "E_p", "E_d", "E_d/N_a", "Fmax_v", "Fmax_p", "Disp")
        + ("{:>6s}  " * 2).format("Erk_v", "Erk_p")
        + "\n"
    )

    num_structures = len(natoms)

    indices = np.arange(num_structures, dtype=np.int64)
    sort = np.argsort(ene[0])
    v_rankings = sorted(indices, key=lambda i: sort[i])
    sort = np.argsort(ene[1])
    p_rankings = sorted(indices, key=lambda i: sort[i])

    for i in range(num_structures):
        ene_diff = ene[0][i] - ene[1][i]
        data = [
            ene[0][i],
            ene[1][i],
            ene_diff,
            ene_diff / natoms[i],
            maxfrc[0][i],
            maxfrc[1][i],
            disp[i],
            v_rankings[i],
            p_rankings[i],
        ]
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

        if worker is not None:
            v_worker = worker
            self._print("Use the worker at run time.")
        else:
            v_worker = self.worker
        assert v_worker is not None, "Worker must be provided either init or run time."
        v_worker.directory = self.directory / "_run"

        if structures is not None:
            v_structures = structures
            self._print("Use the structures at run time.")
        else:
            v_structures = self.structures
        self._print(f"{v_structures=}")
        assert v_structures is not None, "Structures must be provided either init or run time."

        if isinstance(v_structures, StructureBuilder):
            v_structures = v_structures.run()

        is_finished = False

        end_frames = self._irun(v_structures, v_worker)
        if end_frames is not None:
            results = compare_structures(v_structures, end_frames, energy_references=self.reference_energies)
            if not pathlib.Path(self.directory / "v.dat").exists():
                if not self.show_ranking:
                    content = summarise_validation(**results)
                else:
                    content = summarise_validation_with_ranking(**results)
                with open(self.directory / "v.dat", "w") as fopen:
                    fopen.write(content)
        else:
            is_finished = False

        return is_finished

    def _irun(self, frames: list[Atoms], worker: DriverBasedWorker) -> Optional[list[Atoms]]:
        """"""
        assert isinstance(worker, DriverBasedWorker), "Worker must be a DriverBasedWorker."

        cache_fpath = self.directory / "pred.xyz"
        if cache_fpath.exists():
            end_frames = read(cache_fpath, ":")
            return end_frames

        _ = worker.run(frames)
        _ = worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            trajectories = worker.retrieve(include_retrieved=True)
            end_frames = [t[-1] for t in trajectories]
            write(cache_fpath, end_frames)
        else:
            end_frames = None

        return end_frames  # type: ignore


if __name__ == "__main__":
    ...
