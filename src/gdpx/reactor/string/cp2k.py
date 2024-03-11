#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import dataclasses
import os
import pathlib
import shutil
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
from .. import parse_constraint_info


def run_cp2k(name, command, directory):
    """Run vasp from the command. 
    
    ASE Vasp does not treat restart of a MD simulation well. Therefore, we run 
    directly from the command if INCAR aready exists.
    
    """
    import subprocess
    from ase.calculators.calculator import EnvironmentError, CalculationFailed

    try:
        proc = subprocess.Popen(command, shell=True, cwd=directory)
    except OSError as err:
        # Actually this may never happen with shell=True, since
        # probably the shell launches successfully.  But we soon want
        # to allow calling the subprocess directly, and then this
        # distinction (failed to launch vs failed to run) is useful.
        msg = 'Failed to execute "{}"'.format(command)
        raise EnvironmentError(msg) from err

    errorcode = proc.wait()

    if errorcode:
        path = os.path.abspath(directory)
        msg = ('Calculator "{}" failed with command "{}" failed in '
               '{} with error code {}'.format(name, command,
                                              path, errorcode))
        raise CalculationFailed(msg)

    return


@dataclasses.dataclass
class Cp2kStringReactorSetting(StringReactorSetting):

    backend: str = "cp2k"

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
                ("MOTION/BAND", f"K_SPRING {self.k/(units.Hartree/units.Bohr**2)}"),
                ("MOTION/BAND", "ROTATE_FRAMES F"),
                ("MOTION/BAND", "ALIGN_FRAMES F"),
                ("MOTION/BAND/CI_NEB", "NSTEPS_IT 2"),
                ("MOTION/BAND/OPTIMIZE_BAND", "OPT_TYPE DIIS"),
                ("MOTION/BAND/OPTIMIZE_BAND/DIIS", "NO_LS T"),
                ("MOTION/BAND/OPTIMIZE_BAND/DIIS", "N_DIIS 3"),
                ("MOTION/PRINT/RESTART_HISTORY/EACH", f"BAND {self.ckpt_period}"),
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
        rmax_ = kwargs.get("rmax", self.rmax)
        rrms_ = kwargs.get("rrms", self.rrms)
        fmax_ = kwargs.get("fmax", self.fmax)
        frms_ = kwargs.get("frms", self.frms)
        steps_ = kwargs.get("steps", self.steps)

        run_pairs = []
        run_pairs.append(
            ("MOTION/BAND/OPTIMIZE_BAND/DIIS", f"MAX_STEPS {steps_}")
        )
        if fmax_ is not None:
            run_pairs.extend(
                [
                    ("MOTION/BAND/CONVERGENCE_CONTROL", f"MAX_FORCE {fmax_/(units.Hartree/units.Bohr)}"),
                    ("MOTION/BAND/CONVERGENCE_CONTROL", f"MAX_DR {rmax_/(units.Bohr)}"),
                    ("MOTION/BAND/CONVERGENCE_CONTROL", f"RMS_FORCE {frms_/(units.Hartree/units.Bohr)}"),
                    ("MOTION/BAND/CONVERGENCE_CONTROL", f"RMS_DR {rrms_/(units.Bohr)}"),
                ]
            )

        # - add constraint
        run_params = dict(
            constraint = kwargs.get("constraint", self.constraint),
            run_pairs = run_pairs
        )

        return run_params


class Cp2kStringReactor(AbstractStringReactor):

    name: str = "cp2k"

    traj_name: str = "cp2k.out"

    def __init__(self, calc=None, params={}, ignore_convergence=False, directory="./", *args, **kwargs) -> None:
        """"""
        self.calc = calc
        if self.calc is not None:
            self.calc.reset()

        self.ignore_convergence = ignore_convergence

        self.directory = directory

        # - parse params
        self.setting = Cp2kStringReactorSetting(**params)
        self._debug(self.setting)

        return
    
    def _verify_checkpoint(self):
        """Check if the current directory has any valid outputs or 
            it just created the input files.

        """
        verified = super()._verify_checkpoint()
        if verified:
            checkpoints = list(self.directory.glob("*.restart"))
            self._debug(f"checkpoints: {checkpoints}")
            if not checkpoints:
                verified = False
        else:
            ...

        return verified
    
    def _irun(self, structures: List[Atoms], ckpt_wdir=None, *args, **kwargs):
        """"""
        try:
            if ckpt_wdir is None: # start from the scratch
                images = self._align_structures(structures)
                write(self.directory/"images.xyz", images)
                atoms = images[0] # use the initial state

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
            else: # start from a checkpoint
                atoms  = read(ckpt_wdir/"images.xyz", "0")
                with open(ckpt_wdir/"cp2k.inp", "r") as fopen:
                    inp = "".join(fopen.readlines())
                sec = parse_input(inp)

                def remove_keyword(section, keywords):
                    """"""
                    for kw in keywords:
                        parts = kw.upper().split("/")
                        subsection = section.get_subsection("/".join(parts[0:-1]))
                        new_subkeywords = []
                        for subkw in subsection.keywords:
                            if parts[-1] not in subkw:
                                new_subkeywords.append(subkw)
                        subsection.keywords = new_subkeywords

                    return
                
                remove_keyword(
                    sec, 
                    keywords=[ # avoid conflicts
                        "GLOBAL/PROJECT", "GLOBAL/PRINT_LEVEL", "FORCE_EVAL/METHOD", 
                        "FORCE_EVAL/DFT/BASIS_SET_FILE_NAME", "FORCE_EVAL/DFT/POTENTIAL_FILE_NAME",
                        "FORCE_EVAL/SUBSYS/CELL/PERIODIC", 
                        "FORCE_EVAL/SUBSYS/CELL/A", "FORCE_EVAL/SUBSYS/CELL/B", "FORCE_EVAL/SUBSYS/CELL/C"
                    ]
                )

                sec.add_keyword("EXT_RESTART/RESTART_FILE_NAME", str(ckpt_wdir/"cp2k-1.restart"))
                sec.add_keyword("FORCE_EVAL/DFT/SCF/SCF_GUESS", "RESTART")

                # - copy wavefunctions...
                restart_wfns = sorted(list(ckpt_wdir.glob("*.wfn")))
                for wfn in restart_wfns:
                    (self.directory/wfn.name).symlink_to(wfn, target_is_directory=False)

            # - update input
            self.calc.parameters.inp = "\n".join(sec.write())
            atoms.calc = self.calc

            # - run calculation
            self.calc.atoms = atoms
            self.calc.write_input(atoms)
            run_cp2k("cp2k", self.calc.command, self.directory)
            self.calc.atoms = None

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
    
    def _read_a_single_trajectory(self, wdir, *args, **kwargs):
        """

        NOTE: Fixed atoms have zero forces.

        """
        self._debug(f"***** read_trajectory *****")
        self._debug(f"{str(wdir)}")
        cell = None # TODO: if no pbc?
        natoms = None
        nimages = None
        temp_forces, temp_energies = [], []
        energies, forces = [], []
        with open(wdir/"cp2k.out", "r") as fopen:
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
                    # line = fopen.readline() # BUG: inconsistent Images and Replicas?
                    if not line:
                        break
                    try:
                        nimages = int(line.strip().split()[-2])
                        self._debug(line)
                        self._debug(f"nimages: {nimages}")
                    except Exception as e:
                        self._debug("nimages is not found.")
                # NOTE: For method with LineSearch, several SCF may be performed at one step
                """
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
                """
                if "Computing Energies and Forces" in line:
                    # NEB| REPLICA Nr.    1- Energy and Forces
                    # NEB|                                     Total energy:       -2940.286865478840
                    # NEB|    ATOM                            X                Y                Z
                    curr_data = []
                    found_replica_forces = False
                    for i in range((natoms+3)*nimages):
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
                            float(curr_data[i].strip().split()[-1]) for i in range(1, len(curr_data), natoms+3)
                        ]
                        temp_energies.append(curr_energies)
                        curr_forces = []
                        for ir in range(nimages):
                            curr_forces.append(
                                [curr_data[i].strip().split()[2:] for i in range((natoms+3)*ir+3, (natoms+3)*ir+3+natoms)]
                            )
                        temp_forces.append(curr_forces)
                    else:
                        break
                if "BAND TOTAL ENERGY" in line:
                    if temp_energies and temp_forces: # if the step completed...
                        #print("temp_energies: ", len(temp_energies))
                        #print("temp_forces: ", np.array(temp_forces, dtype=np.float64).shape)
                        energies.append(temp_energies[-1])
                        forces.extend(temp_forces[-1])
                        temp_forces, temp_energies = [], []

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
            if nimages < 10:
                for i in range(nimages):
                    curr_xyzfile = wdir/f"cp2k-pos-Replica_nr_{i+1}-1.xyz"
                    curr_frames = read(curr_xyzfile, index=":", format="xyz")[:nbands]
                    frames_.append(curr_frames)
            else:
                for i in range(nimages):
                    curr_xyzfile = wdir/f"cp2k-pos-Replica_nr_{str(i+1).zfill(2)}-1.xyz"
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
