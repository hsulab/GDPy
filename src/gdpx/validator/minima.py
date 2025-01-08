#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from typing import Any, Optional

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.io import read, write

from gdpx.builder.builder import StructureBuilder
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
        pbc=copy.deepcopy(atoms_.get_pbc())
    )
    if results is not None:
        spc = SinglePointCalculator(atoms, **results)
        atoms.calc = spc

    return atoms


def compare_structures(v_frames: list[Atoms], p_frames: list[Atoms]):
    """"""
    # number of atoms
    v_natoms = np.array([len(a) for a in v_frames])
    p_natoms = np.array([len(a) for a in p_frames])
    assert np.allclose(v_natoms, p_natoms), "Number of atoms are not consistent."

    # total energies
    v_ene = np.array([a.get_potential_energy() for a in v_frames])
    p_ene = np.array([a.get_potential_energy() for a in p_frames])

    # maximum forces TODO: constraints?
    v_maxfrc = np.array([np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in v_frames])
    p_maxfrc = np.array([np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in p_frames])

    # displacement
    disp = [] # displacements
    for ref_atoms, pre_atoms in zip(v_frames, p_frames):
        vector = pre_atoms.get_positions() - ref_atoms.get_positions()
        _, vlen = find_mic(vector, pre_atoms.get_cell())
        disp.append(np.linalg.norm(vlen))

    results = dict(
        natoms = v_natoms,
        ene = (v_ene, p_ene),
        maxfrc = (v_maxfrc, p_maxfrc),
        disp = disp
    )

    return results


def summarise_validation(natoms, ene, maxfrc, disp) -> str:
    """"""
    line_format = "{:>6d}  " * 2 + "{:>12.4f}  " * 6 + "\n"

    content = "# Name     N_a  " + ("{:>12s}  "*6).format("E_v", "E_p", "E_d", "E_d/N_a", "Fmax_v", "Fmax_p", "Disp") + "\n"

    num_structures = len(natoms)
    for i in range(num_structures):
        ene_diff = ene[0][i] - ene[1][i]
        data = [ene[0][i], ene[1][i], ene_diff, ene_diff/natoms[i], maxfrc[0][i], maxfrc[1][i], disp[i]]
        content += line_format.format(i, natoms[i], *data)

    return content


class MinimaValidator(BaseValidator):

    """Run minimisation on various configurations and compare relative energy.

    TODO: 

        Support the comparison of minimisation trajectories.

    """

    name: str = "minima"

    def __init__(self, ene_shift=[], *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.ene_shift = ene_shift

        return

    def run(self, structures: Optional[Any]=None, worker: Optional[DriverBasedWorker]=None, *args, **kwargs):
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
            results = compare_structures(v_structures, end_frames)
            if not pathlib.Path(self.directory / "v.dat").exists():
                content = summarise_validation(**results)
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
