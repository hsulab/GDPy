#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import dataclasses
import pathlib
from typing import Callable, List

import numpy as np

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    print("Used default matplotlib style.")

from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic
from ase.constraints import Filter, FixAtoms
from ase.neb import NEB, NEBTools
from ase.calculators.singlepoint import SinglePointCalculator


from .. import config as GDPCONFIG
from ..computation.mixer import EnhancedCalculator
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


def compute_rxn_coords(frames):
    """Compute reaction coordinates."""
    rxn_coords = []

    cell = frames[0].get_cell(complete=True)
    nframes = len(frames)
    for i in range(1,nframes):
        prev_positions = frames[i-1].get_positions()
        curr_positions = frames[i].get_positions()
        shift = curr_positions - prev_positions
        curr_vectors, curr_distances = find_mic(shift, cell, pbc=True)
        frames[i].positions = prev_positions + curr_vectors
        rxn_coords.append(np.linalg.norm(curr_vectors))

    rxn_coords = np.cumsum(rxn_coords)
    rxn_coords = np.hstack(([0.],rxn_coords))

    return rxn_coords

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


def plot_mep(wdir, images):
    """"""
    print("nimages: ", len(images))
    rxn_coords = compute_rxn_coords(images)
    print("rxn_coords: ", rxn_coords)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    plt.suptitle("Nudge Elastic Band Calculation")

    nbt = NEBTools(images=images)
    nbt.plot_band(ax=ax)

    plt.savefig(wdir/"neb.png")

    return


@dataclasses.dataclass
class ReactorSetting:

    nimages: int = 7
    
    #:
    optimiser: str = "bfgs"

    #: Convergence force tolerance.
    fmax: float = 0.05

    #: Maximum number of steps.
    steps: int = 100

    #: FixAtoms.
    constraint: str = None

    def __post_init__(self):
        """"""
        # - ...
        opt_cls = None
        if self.optimiser == "bfgs":
            from ase.optimize import BFGS
            opt_cls = BFGS
        
        self.opt_cls = opt_cls

        return

    def get_run_params(self):
        """"""
        run_params = dict(
            steps = self.steps,
            fmax = self.fmax
        )

        return run_params


class MEPFinder():

    """Find the minimum energy path based on input structures.

    Methods based on the number of input structures such as single, double, multi...

    """

    name = "ase"

    #: Standard print.
    _print: Callable = GDPCONFIG._print

    #: Standard debug.
    _debug: Callable = GDPCONFIG._debug

    _directory = "./"

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
        self.setting = ReactorSetting(**params)
        self._debug(self.setting)

        return

    @property
    def directory(self):
        """Set working directory of this driver.

        Note:
            The attached calculator's directory would be set as well.
        
        """

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        self._directory = pathlib.Path(directory_)
        self.calc.directory = str(self.directory) # NOTE: avoid inconsistent in ASE

        self.cache_nebtraj = self.directory/self.traj_name

        return
    
    def run(self, structures: List[Atoms], read_cache=True, *args, **kwargs):
        """"""
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
            ...
        
        _ = self.read_trajectory()

        return
    
    def _irun(self, structures):
        """"""
        # - check lattice consistency
        ini_atoms, fin_atoms = structures
        c1, c2 = ini_atoms.get_cell(complete=True), fin_atoms.get_cell(complete=True)
        assert np.allclose(c1, c2), "Inconsistent unit cell..."

        # - align structures
        shift = fin_atoms.get_positions() - ini_atoms.get_positions()
        curr_vectors, curr_distances = find_mic(shift, c1, pbc=True)
        self._print(f"curr_vectors: {curr_vectors}")
        self._print(f"disp: {np.linalg.norm(curr_vectors)}")
        fin_atoms.positions = ini_atoms.get_positions() + curr_vectors

        ini_atoms = set_constraint(ini_atoms, self.setting.constraint)
        fin_atoms = set_constraint(fin_atoms, self.setting.constraint)

        # - find mep
        nimages = self.setting.nimages
        images = [ini_atoms]
        images += [ini_atoms.copy() for i in range(nimages-2)]
        images.append(fin_atoms)

        for a in images:
            a.calc = self.calc

        neb = NEB(
            images, allow_shared_calculator=True, climb=False, k=0.1
        )
        neb.interpolate(apply_constraint=True)

        #qn = self.setting.opt_cls(neb, logfile="-", trajectory=str(self.directory/"nebtraj.traj"))
        qn = self.setting.opt_cls(neb, logfile="-", trajectory=None)
        qn.attach(
            save_nebtraj, interval=1,
            neb=neb, log_fpath=self.cache_nebtraj
        )

        run_params = self.setting.get_run_params()
        qn.run(steps=run_params["steps"], fmax=run_params["fmax"])

        return
    
    def read_trajectory(self, *args, **kwargs):
        """"""
        if self.cache_nebtraj.exists():
            images = read(self.cache_nebtraj, ":")
            plot_mep(self.directory, images[-self.setting.nimages:])
        else:
            raise FileNotFoundError(f"No cache trajectory {str(self.cache_nebtraj)}.")

        return
    
    def read_convergence(self, *args, **kwargs):
        """"""

        return


if __name__ == "__main__":
    ...