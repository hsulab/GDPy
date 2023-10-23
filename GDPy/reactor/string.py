#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import pathlib
from typing import List

import numpy as np

from ase import Atoms
from ase.geometry import find_mic
from ase.constraints import FixAtoms
from ase.neb import interpolate, idpp_interpolate

from .reactor import AbstractReactor
from GDPy.builder.constraints import parse_constraint_info


def set_constraint(atoms, cons_text):
    """"""
    atoms._del_constraints()
    mobile_indices, frozen_indices = parse_constraint_info(
        atoms, cons_text, ignore_ase_constraints=True, ret_text=False
    )
    if frozen_indices:
        atoms.set_constraint(FixAtoms(indices=frozen_indices))

    return atoms


@dataclasses.dataclass
class StringReactorSetting:

    #: Period to save the restart file.
    restart_period: int = 100

    #: Number of images along the pathway.
    nimages: int = 7

    #: Align IS and FS based on the mic.
    mic: bool = True
    
    #: Optimiser.
    optimiser: str = "bfgs"

    #: Spring constant, eV/Ang2.
    k: float = 0.1

    #: Whether use CI-NEB.
    climb: bool = False

    #: Convergence force tolerance.
    fmax: float = 0.05

    #: Maximum number of steps.
    steps: int = 100

    #: FixAtoms.
    constraint: str = None

    #: Parameters that are used to update.
    _internals: dict = dataclasses.field(default_factory=dict)

    def get_init_params(self):
        """"""

        return copy.deepcopy(self._internals)

    def get_run_params(self):
        """"""

        raise NotImplementedError(f"{self.__class__.__name__} has no function for run params.")

class AbstractStringReactor(AbstractReactor):

    name: str = "string"

    traj_name: str = "nebtraj.xyz"

    @AbstractReactor.directory.setter
    def directory(self, directory_):
        self._directory = pathlib.Path(directory_)
        self.calc.directory = str(self.directory) # NOTE: avoid inconsistent in ASE

        self.cache_nebtraj = self.directory/self.traj_name

        return

    def run(self, structures: List[Atoms], read_cache=True, *args, **kwargs):
        """"""
        super().run(structures=structures, *args, **kwargs)

        # - Double-Ended Methods...
        ini_atoms, fin_atoms = structures
        self._print(f"ini_atoms: {ini_atoms.get_potential_energy()}")
        self._print(f"fin_atoms: {fin_atoms.get_potential_energy()}")

        # - backup old parameters
        prev_params = copy.deepcopy(self.calc.parameters)

        # -
        if not self.cache_nebtraj.exists():
            self._irun([ini_atoms, fin_atoms])
        else:
            # - check if converged
            converged = self.read_convergence()
            if not converged:
                if read_cache:
                    ...
                self._irun(structures, *args, **kwargs)
            else:
                ...
        
        #self.calc.set(**prev_params)
        
        # - get results
        _ = self.read_trajectory()

        return

    def _align_structures(self, structures, *args, **kwargs) -> List[Atoms]:
        """"""
        nstructures = len(structures)
        if nstructures == 2:
            # - check lattice consistency
            ini_atoms, fin_atoms = structures
            c1, c2 = ini_atoms.get_cell(complete=True), fin_atoms.get_cell(complete=True)
            assert np.allclose(c1, c2), "Inconsistent unit cell..."

            # - align structures
            shifts = fin_atoms.get_positions() - ini_atoms.get_positions()
            if self.setting.mic:
                self._print("Align IS and FS based on MIC.")
                curr_vectors, curr_distances = find_mic(shifts, c1, pbc=True)
                self._debug(f"curr_vectors: {curr_vectors}")
                self._print(f"disp: {np.linalg.norm(curr_vectors)}")
                fin_atoms.positions = ini_atoms.get_positions() + curr_vectors
            else:
                self._print(f"disp: {np.linalg.norm(shifts)}")

            ini_atoms = set_constraint(ini_atoms, self.setting.constraint)
            fin_atoms = set_constraint(fin_atoms, self.setting.constraint)

            # - find mep
            nimages = self.setting.nimages
            images = [ini_atoms]
            images += [ini_atoms.copy() for i in range(nimages-2)]
            images.append(fin_atoms)

            interpolate(
                images=images, mic=False, interpolate_cell=False, 
                use_scaled_coord=False, apply_constraint=None
            )
        else:
            self._print("Use a pre-defined pathway.")
            images = [a.copy() for a in structures]

        return images
    
    def as_dict(self) -> dict:
        """"""
        params = {}

        return params


if __name__ == "__main__":
    ...