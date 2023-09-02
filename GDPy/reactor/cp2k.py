#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import dataclasses
import pathlib
import traceback

from typing import Union, List

import numpy as np

from ase import Atoms
from ase import units
from ase.io import read, write
from ase.calculators.cp2k import parse_input, InputSection
from ase.calculators.singlepoint import SinglePointCalculator
from ase.neb import NEB

from .string import AbstractStringReactor, StringReactorSetting
from ..builder.constraints import parse_constraint_info
from .utils import plot_bands, plot_mep


@dataclasses.dataclass
class Cp2kStringReactorSetting(StringReactorSetting):

    #: Number of tasks/processors/cpus for each image.
    ntasks_per_image: int = 1

    def __post_init__(self):
        """"""
        pairs = []

        pairs.extend(
            [
                ("GLOBAL", "RUN_TYPE BAND"),
                ("MOTION/BAND", "BAND_TYPE CI-NEB"),
                ("MOTION/BAND", f"NPROC_REP {self.ntasks_per_image}"),
                ("MOTION/BAND", f"NUMBER_OF_REPLICA {self.nimages}"),
                ("MOTION/BAND", f"K_SPRING {self.k}"),
                ("MOTION/BAND", "ROTATE_FRAMES F"),
                ("MOTION/BAND", "ALIGN_FRAMES F"),
                ("MOTION/BAND/CI_NEB", "NSTEPS_IT 2"),
                ("MOTION/BAND/OPTIMIZE_BAND", "OPT_TYPE DIIS"),
                ("MOTION/PRINT/RESTART_HISTORY/EACH", f"BAND {self.restart_period}"),
            ]
        )

        pairs.extend(
            [
                ("MOTION/PRINT/CELL", "_SECTION_PARAMETERS_ ON"),
                ("MOTION/PRINT/TRAJECTORY", "_SECTION_PARAMETERS_ ON"),
                ("MOTION/PRINT/FORCES", "_SECTION_PARAMETERS_ ON"),
            ]
        )
        self._internals["pairs"] = pairs

        return
    
    def get_run_params(self, *args, **kwargs):
        """"""
        # - convergence criteria
        fmax_ = kwargs.get("fmax", self.fmax)
        steps_ = kwargs.get("steps", self.steps)

        run_pairs = []
        run_pairs.append(
            ("MOTION/BAND/OPTIMIZE_BAND/DIIS", f"MAX_STEPS {self.steps}")
        )
        if fmax_ is not None:
            run_pairs.append(
                ("MOTION/BAND/CONVERGENCE_CONTROL", f"MAX_FORCE {fmax_/(units.Hartree/units.Bohr)}")
            )

        # - add constraint
        run_params = dict(
            constraint = kwargs.get("constraint", self.constraint),
            run_pairs = run_pairs
        )

        return run_params


class Cp2kStringReactor(AbstractStringReactor):

    name: str = "cp2k"

    def __init__(self, calc=None, params={}, ignore_convergence=False, directory="./", *args, **kwargs) -> None:
        """"""
        self.calc = calc
        if self.calc is not None:
            self.calc.reset()

        self.ignore_convergence = ignore_convergence

        self.directory = directory
        self.cache_nebtraj = self.directory/self.traj_name

        # - parse params
        self.setting = Cp2kStringReactorSetting(**params)
        self._debug(self.setting)

        return
    
    def _verify_checkpoint(self):
        """Check if the current directory has any valid outputs or it just created 
            the input files.

        """
        checkpoints = list(self.directory.glob("*.restart"))
        print(f"checkpoints: {checkpoints}")

        return checkpoints

    def run(self, structures: List[Atoms], read_cache=True, *args, **kwargs):
        """"""
        #super().run(structures=structures, *args, **kwargs)

        # - Double-Ended Methods...
        ini_atoms, fin_atoms = structures
        self._print(f"ini_atoms: {ini_atoms.get_potential_energy()}")
        self._print(f"fin_atoms: {fin_atoms.get_potential_energy()}")

        # - backup old parameters
        prev_params = copy.deepcopy(self.calc.parameters)
        print(f"prev_params: {prev_params}")

        # -
        if not self._verify_checkpoint(): # is not a []
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
        
        self.calc.set(**prev_params)
        
        # - get results
        band_frames = self.read_trajectory() # (nbands, nimages)
        if band_frames:
            plot_mep(self.directory, band_frames[-1])
            #plot_bands(self.directory, images, nimages=nimages_per_band)
            write(self.directory/"nebtraj.xyz", itertools.chain(*band_frames))
            # --
            last_band = band_frames[-1]
            energies = [a.get_potential_energy() for a in last_band]
            imax = 1 + np.argsort(energies[1:-1])[-1]
            print(f"imax: {imax}")
            maxfrc = np.max(last_band[imax].get_forces(apply_constraint=True))
            print(f"maxfrc: {maxfrc}")
        else:
            last_band = []

        return last_band
    
    def _irun(self, structures: List[Atoms], *args, **kwargs):
        """"""
        images = self._align_structures(structures)
        write(self.directory/"images.xyz", images)

        atoms = images[0] # use the initial state
        try:
            run_params = self.setting.get_run_params(**kwargs)
            run_params.update(**self.setting.get_init_params())

            # - update input template
            # GLOBAL section is automatically created...
            # FORCE_EVAL.(METHOD, POISSON)
            inp = self.calc.parameters.inp # string
            sec = parse_input(inp)
            for (k, v) in run_params["pairs"]:
                sec.add_keyword(k, v)
            for (k, v) in run_params["run_pairs"]:
                sec.add_keyword(k, v)

            # -- check constraint
            cons_text = run_params.pop("constraint", None)
            mobile_indices, frozen_indices = parse_constraint_info(
                atoms, cons_text=cons_text, ignore_ase_constraints=True, ret_text=False
            )
            if frozen_indices:
                #atoms._del_constraints()
                #atoms.set_constraint(FixAtoms(indices=frozen_indices))
                frozen_indices = sorted(frozen_indices)
                sec.add_keyword(
                    "MOTION/CONSTRAINT/FIXED_ATOMS", 
                    "LIST {}".format(" ".join([str(i+1) for i in frozen_indices]))
                )
            
            # -- add replica information
            band_section = sec.get_subsection("MOTION/BAND")
            for replica in images:
                cur_rep = InputSection(name="REPLICA")
                for pos in replica.positions:
                    cur_rep.add_keyword("COORD", ("{:.18e} "*3).format(*pos), unique=False)
                band_section.subsections.append(cur_rep)

            # - update input
            self.calc.parameters.inp = "\n".join(sec.write())
            atoms.calc = self.calc

            # - run calculation
            _ = atoms.get_forces()

        except Exception as e:
            self._debug(e)
            self._debug(traceback.print_exc())

        return
    
    def read_convergence(self, *args, **kwargs):
        """"""
        converged = super().read_convergence(*args, **kwargs)

        with open(self.directory/"cp2k.out", "r") as fopen:
            lines = fopen.readlines()
        
        for line in lines:
            if "PROGRAM ENDED AT" in line:
                converged = True
                break

        return converged
    
    def read_trajectory(self, *args, **kwargs):
        """

        NOTE: Fixed atoms have zero forces.

        """
        self._debug(f"***** read_trajectory *****")
        cell = None # TODO: if no pbc?
        natoms = None
        nimages = None
        energies, forces = [], []
        with open(self.directory/"cp2k.out", "r") as fopen:
            while True:
                line = fopen.readline()
                if not line:
                    break
                # - find cell
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
                        except Exception as e:
                            self._debug("cell is not found.")
                            break
                # - find natoms
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
                            natoms = int(line.strip().split()[-1])
                            self._debug(f"natoms: {natoms}")
                        except Exception as e:
                            self._debug("natoms is not found.")
                            break
                    else:
                        break
                if "Number of Images" in line:
                    line = fopen.readline()
                    if not line:
                        break
                    try:
                        nimages = int(line.strip().split()[-2])
                        self._debug(f"nimages: {nimages}")
                    except Exception as e:
                        self._debug("nimages is not found.")
                #if "Computing Energies and Forces" in line:
                if "REPLICA Nr." in line:
                    assert natoms is not None and nimages is not None, f"natoms: {natoms}, nimages: {nimages}"
                    curr_data = []
                    found_replica_forces = False
                    for i in range(natoms+2):
                        line = fopen.readline()
                        if line:
                            curr_data.append(line)
                        else:
                            break
                    else:
                        # current replica's forces are complete...
                        found_replica_forces = True
                    if found_replica_forces:
                        curr_energy = float(curr_data[0].strip().split()[-1])
                        energies.append(curr_energy)
                        curr_forces = [x.strip().split()[2:] for x in curr_data[2:]]
                        forces.append(curr_forces)
                    else:
                        break

        # - truncate to the last complete band
        frames = [] # shape (nbands, nimages)
        if forces:
            forces = np.array(forces, dtype=np.float64)
            shape = forces.shape
            self._debug(f"forces: {shape}")
            nbands = int(shape[0]/nimages)
            forces = forces[:nbands*nimages]
            self._debug(f"truncated forces: {forces.shape} nbands: {nbands}")
            forces = np.reshape(forces, (nbands, nimages, natoms, -1)) # shape (nbands, nimages, natoms, 3)
            forces *= units.Hartree/units.Bohr

            energies = np.array(energies)[:nbands*nimages].reshape(nbands, nimages)
            energies *= units.Hartree
            self._debug(f"energies: {energies.shape} nbands: {nbands}")

            cell = np.array(cell, dtype=np.float64)
            self._debug(f"cell: {cell}")

            # - read positions
            frames_ = [] # shape (nimages, nbands)
            for i in range(nimages):
                curr_xyzfile = self.directory/f"cp2k-pos-Replica_nr_{i+1}-1.xyz"
                curr_frames = read(curr_xyzfile, index=":", format="xyz")[:nbands]
                frames_.append(curr_frames)
            for j in range(nbands):
                curr_band = []
                for i in range(nimages):
                    curr_band.append(frames_[i][j])
                frames.append(curr_band)
            for i in range(nbands):
                for j in range(nimages):
                    atoms = frames[i][j]
                    atoms.set_cell(cell)
                    atoms.pbc = True
                    spc = SinglePointCalculator(
                        atoms, energy=energies[i, j], free_energy=energies[i, j],
                        forces=forces[i, j].copy()
                    )
                    atoms.calc = spc
        else:
            ...

        return frames


if __name__ == "__main__":
    ...