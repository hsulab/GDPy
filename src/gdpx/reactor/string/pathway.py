#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import dataclasses
import pathlib
from typing import Callable, List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic
from ase.constraints import Filter, FixAtoms
from ase.neb import NEB, NEBTools
from ase.calculators.singlepoint import SinglePointCalculator


from .. import config as GDPCONFIG
from .. import EnhancedCalculator
from .. import parse_constraint_info
from .string import AbstractStringReactor, StringReactorSetting

def set_constraint(atoms, cons_text):
    """"""
    atoms._del_constraints()
    mobile_indices, frozen_indices = parse_constraint_info(
        atoms, cons_text, ignore_ase_constraints=True, ret_text=False
    )
    if frozen_indices:
        atoms.set_constraint(FixAtoms(indices=frozen_indices))

    return atoms

def print_step(neb, pfunc):
    """"""
    content = "{} {}"

    return

def save_nebtraj(neb, log_fpath) -> None:
    """Create a clean atoms from the input and save simulation trajectory.

    We need an explicit copy of atoms as some calculators may not return all 
    necessary information. For example, schnet only returns required properties.
    If only energy is required, there are no forces.

    """
    def _convert_atoms(atoms):
        # - save atoms
        atoms_to_save = Atoms(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions().copy(),
            cell=atoms.get_cell().copy(),
            pbc=copy.deepcopy(atoms.get_pbc())
        )
        if "tags" in atoms.arrays:
            atoms_to_save.set_tags(atoms.get_tags())
        if atoms.get_kinetic_energy() > 0.:
            atoms_to_save.set_momenta(atoms.get_momenta())
        results = dict(
            energy = atoms.get_potential_energy(),
            forces = copy.deepcopy(atoms.get_forces())
        )
        spc = SinglePointCalculator(atoms, **results)
        atoms_to_save.calc = spc

        # - save special keys and arrays from calc
        natoms = len(atoms)
        # -- add deviation
        for k, v in atoms.calc.results.items():
            if k in GDPCONFIG.VALID_DEVI_FRAME_KEYS:
                atoms_to_save.info[k] = v
        for k, v in atoms.calc.results.items():
            if k in GDPCONFIG.VALID_DEVI_ATOMIC_KEYS:
                atoms_to_save.arrays[k] = np.reshape(v, (natoms, -1))

        # -- check special metadata
        calc = atoms.calc
        if isinstance(calc, EnhancedCalculator):
            atoms_to_save.info["host_energy"] = copy.deepcopy(calc.results["host_energy"])
            atoms_to_save.arrays["host_forces"] = copy.deepcopy(calc.results["host_forces"])

        return atoms_to_save
    
    # - append to traj
    for atoms in neb.iterimages():
        atoms_to_save = _convert_atoms(atoms)
        write(log_fpath, atoms_to_save, append=True)

    return


@dataclasses.dataclass
class AseStringReactorSetting(StringReactorSetting):

    backend: str = "ase"

    def __post_init__(self):
        """"""
        # - ...
        opt_cls = None
        if self.optimiser == "bfgs": # Takes a lot of time to solve hessian.
            from ase.optimize import BFGS as opt_cls
        elif self.optimiser == "fire": # BUG: STRANGE BEHAVIOUR.
            from ase.optimize import FIRE as opt_cls
        elif self.optimiser == "mdmin":
            from ase.optimize import MDMin as opt_cls
        else:
            ...
        
        self.opt_cls = opt_cls

        return

    def get_run_params(self, *args, **kwargs):
        """"""
        steps_ = kwargs.get("steps", self.steps)
        if steps_ <= 0:
            steps_ = -1
        run_params = dict(
            steps = steps_,
            fmax = self.fmax
        )

        return run_params


class AseStringReactor(AbstractStringReactor):

    """Find the minimum energy path based on input structures.

    Methods based on the number of input structures such as single, double, multi...

    """

    name = "ase"

    traj_name: str = "nebtraj.xyz"

    def __init__(self, calc=None, params={}, ignore_convergence=False, directory="./", *args, **kwargs) -> None:
        """"""
        self.calc = calc
        if self.calc is not None:
            self.calc.reset()

        self.ignore_convergence = ignore_convergence

        self.directory = directory
        self.cache_nebtraj = self.directory/self.traj_name

        # - parse params
        self.setting = AseStringReactorSetting(**params)
        self._debug(self.setting)

        return
    
    @AbstractStringReactor.directory.setter
    def directory(self, directory_):
        self._directory = pathlib.Path(directory_)
        self.calc.directory = str(self.directory) # NOTE: avoid inconsistent in ASE

        self.cache_nebtraj = self.directory/self.traj_name

        return
    
    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """"""
        verified = super()._verify_checkpoint(*args, **kwargs)
        if verified:
            if self.cache_nebtraj.exists() and self.cache_nebtraj.stat().st_size != 0:
                verified = True
            else:
                verified = False
        else:
            ...

        return verified
    
    def _irun(self, structures: List[Atoms], ckpt_wdir=None, *args, **kwargs):
        """"""
        try:
            run_params = self.setting.get_run_params()

            if ckpt_wdir is None: # start from the scratch
                images = self._align_structures(structures)
                write(self.directory/"images.xyz", images)
            else:
                nebtraj = self.read_trajectory()
                images = nebtraj[-1]
                run_params["steps"] = run_params["steps"] - (len(nebtraj)-1)

            for a in images:
                set_constraint(a, self.setting.constraint)
                a.calc = self.calc

            neb = NEB(
                images=images, k=self.setting.k, climb=self.setting.climb,
                remove_rotation_and_translation=False, method="aseneb",
                allow_shared_calculator=True, precon=None,
            )
            #neb.interpolate(apply_constraint=True)

            driver = self.setting.opt_cls(neb, logfile="-", trajectory=None)
            driver.attach(
                save_nebtraj, interval=1,
                neb=neb, log_fpath=self.cache_nebtraj
            )

            driver.run(steps=run_params["steps"], fmax=run_params["fmax"])
        except Exception as e:
            self._debug(e)

        return
    
    def _read_a_single_trajectory(self, wdir, *args, **kwargs):
        """"""
        cache_nebtraj = wdir/self.traj_name
        nimages_per_band = self.setting.nimages
        if cache_nebtraj.exists():
            images = read(cache_nebtraj, ":")
        else:
            raise FileNotFoundError(f"No cache trajectory {str(cache_nebtraj)}.")
        
        nimages = len(images)
        assert nimages%nimages_per_band == 0, "Inconsistent number of bands."
        nbands = int(nimages/nimages_per_band)

        reshaped_images = []
        for i in range(nbands):
            reshaped_images.append(images[i*nimages_per_band:(i+1)*nimages_per_band])

        return reshaped_images
    
    def read_convergence(self, *args, **kwargs) -> bool:
        """"""
        converged = False
        if self.cache_nebtraj.exists():
            nimages_per_band = self.setting.nimages
            images = self.read_trajectory()
            nsteps = len(images)
            nt = NEBTools(images[-1])
            fmax = nt.get_fmax()
            if (fmax <= self.setting.fmax) or (nsteps >= self.setting.steps+1):
                converged = True
                self._print(
                    f"STEP: {nsteps} >= {self.setting.steps} MAXFRC: {fmax} <=? {self.setting.fmax}"
                )

        return converged


if __name__ == "__main__":
    ...
