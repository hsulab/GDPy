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
from ..data.array import AtomsNDArray
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


def compute_rxn_coords(frames):
    """Compute reaction coordinates."""
    # - avoid change atoms positions and lost energy properties...
    nframes = len(frames)
    natoms = len(frames[0])
    coordinates = np.zeros((nframes, natoms, 3))
    for i, a in enumerate(frames):
        coordinates[i, :, :] = copy.deepcopy(frames[i].get_positions())

    rxn_coords = []
    cell = frames[0].get_cell(complete=True)
    for i in range(1, nframes):
        prev_positions = coordinates[i-1]
        curr_positions = coordinates[i]
        shift = curr_positions - prev_positions
        curr_vectors, curr_distances = find_mic(shift, cell, pbc=True)
        coordinates[i] = prev_positions + curr_vectors
        rxn_coords.append(np.linalg.norm(curr_vectors))

    rxn_coords = np.cumsum(rxn_coords)
    rxn_coords = np.hstack(([0.], rxn_coords))

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


def plot_bands(wdir, images, nimages: int):
    """"""
    #print([a.get_potential_energy() for a in images])
    
    nframes = len(images)

    nbands = int(nframes/nimages)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    plt.suptitle("Nudge Elastic Band Calculation")

    for i in range(nbands):
        #print(f"plot_bands {i}")
        nbt = NEBTools(images=images[i*nimages:(i+1)*nimages])
        nbt.plot_band(ax=ax)

    plt.savefig(wdir/"bands.png")

    return


@dataclasses.dataclass
class ReactorSetting:

    #: Number of images along the pathway.
    nimages: int = 7

    #: Align IS and FS based on the mic.
    mic: bool = True
    
    #: Optimiser.
    optimiser: str = "bfgs"

    #: Spring constant, eV/Ang.
    k: float = 0.1

    #: Whether use CI-NEB.
    climb: bool = False

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

    def get_run_params(self):
        """"""
        run_params = dict(
            steps = self.steps,
            fmax = self.fmax
        )

        return run_params


class MEPFinder(AbstractReactor):

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
        
        # - get results
        _ = self.read_trajectory()

        return
    
    def _irun(self, structures, *args, **kwargs):
        """"""
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

        for a in images:
            a.calc = self.calc

        neb = NEB(
            images=images, k=self.setting.k, climb=self.setting.climb,
            remove_rotation_and_translation=False, method="aseneb",
            allow_shared_calculator=True, precon=None
        )
        neb.interpolate(apply_constraint=True)

        driver = self.setting.opt_cls(neb, logfile="-", trajectory=None)
        driver.attach(
            save_nebtraj, interval=1,
            neb=neb, log_fpath=self.cache_nebtraj
        )

        run_params = self.setting.get_run_params()
        driver.run(steps=run_params["steps"], fmax=run_params["fmax"])

        return
    
    def read_trajectory(self, *args, **kwargs):
        """"""
        nimages_per_band = self.setting.nimages
        if self.cache_nebtraj.exists():
            images = read(self.cache_nebtraj, ":")
            converged_nebtraj = images[-nimages_per_band:]
            #print(self.directory)
            #for a in converged_nebtraj:
            #    print(a.get_potential_energy())
            plot_mep(self.directory, converged_nebtraj)
            plot_bands(self.directory, images, nimages=nimages_per_band)
            write(self.directory/"end_nebtraj.xyz", converged_nebtraj)
        else:
            raise FileNotFoundError(f"No cache trajectory {str(self.cache_nebtraj)}.")
        
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
            images = read(self.cache_nebtraj, ":")
            nimages = len(images)
            nsteps = int(nimages/nimages_per_band)
            end_nebtraj = images[-nimages_per_band:]
            nt = NEBTools(end_nebtraj)
            fmax = nt.get_fmax()
            if (fmax <= self.setting.fmax) or (nsteps >= self.setting.steps):
                converged = True
                self._print(
                    f"STEP: {nsteps} >= {self.setting.steps} MAXFRC: {fmax} <=? {self.setting.fmax}"
                )

        return converged


if __name__ == "__main__":
    ...