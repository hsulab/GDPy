#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" A worker that manages a series of dynamics tasks
    worker needs a machine to dertermine whether run by serial
    or on cluster
"""

import os
import time
import subprocess

import json

from pathlib import Path
from typing import Union, List

import numpy as np

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write
from ase.calculators.vasp import Vasp

from GDPy import config
from GDPy.utils.command import parse_input_file, convert_indices, CustomTimer
from GDPy.calculator.vasp import VaspMachine

from GDPy.machine.machine import AbstractMachine


class VaspWorker():

    """ perform vasp calculations in a directory like
        main
        - vasp0
        - vasp1
        - vasp...
    """

    frames_name = "calculated_0.xyz"
    vasp_dir_pattern = "vasp_*"

    # - attributes
    _machine = None
    _environs = dict(
        VASP_PP_PATH = None,
        ASE_VASP_VDW = None
    )

    def __init__(self, calc):
        """"""
        calc_ = Vasp()
        if isinstance(calc, Vasp):
            calc_ = calc
        else:
            # TODO: check calc type
            calc_.read_json(calc)
            #print("load vasp json: ", calc_.asdict())
        self.calc = calc_

        return
    
    @property
    def machine(self):
        """"""

        return self._machine
    
    @machine.setter
    def machine(self, machine_):
        """"""
        if isinstance(machine_, AbstractMachine):   
            self._machine = machine_
        else:
            raise RuntimeError("Invalid object for worker.machine ...")

        return 
    
    @property
    def environs(self):

        return self._environs
    
    @environs.setter
    def environs(self, environs_):
        """"""
        self._environs = environs_

        return
    
    def set_environs(self):
        """"""
        os.environ["VASP_PP_PATH"] = self.environs["VASP_PP_PATH"]
        os.environ["ASE_VASP_VDW"] = self.environs["ASE_VASP_VDW"]

        return
    
    def _prepare_calculation(self, dpath, extra_params, user_commands):
        """ deal with many calculations in one folder
        """
        # - update system-wise parameters
        self.calc.set(**extra_params)

        calc_params = self.calc.asdict().copy()
        # BUG: ase 3.22.1 no special params in param_state
        calc_params["inputs"]["lreal"] = self.calc.special_params["lreal"] 
        calc_params.update(command = self.calc.command)
        calc_params.update(environs = self.environs)
        with open(dpath / "vasp_params.json", "w") as fopen:
            json.dump(calc_params, fopen, indent=4)

        # -- create job script
        machine_params = {}
        machine_params["job-name"] = dpath.parent.name+"-"+dpath.name+"-fp"

        # TODO: mpirun or mpiexec, move this part to machine object
        #       _parse_resources?
        command = self.calc.command
        if command.strip().startswith("mpirun") or command.strip().startswith("mpiexec"):
            ntasks = command.split()[2]
        else:
            ntasks = 1

        # TODO: number of nodes?
        machine_params.update(
            **{"nodes": "1-1", "ntasks": ntasks, "cpus-per-task": 1, "mem-per-cpu": "4G"}
        )

        self.machine.update(machine_params)

        self.machine.user_commands = user_commands

        self.machine.write(dpath/"vasp.slurm")

        return
    
    def _parse_structures(self, structure_source: Union[str,Path]):
        """ read frames from a list of files
        """
        structure_files = []
        structure_source = Path(structure_source)
        if structure_source.is_file():
            # run calculation 
            #run_calculation(Path.cwd(), args.STRUCTURES, args.indices, vasp_machine, input_dict["command"])
            structure_files.append(structure_source)
        else:
            # all structures share same indices and vasp_machine
            #for sp in structure_source.iterdir():
            #    if sp.is_dir() and (not args.dirs or sp.name in args.dirs):
            #        #p = list(sp.glob("*.xyz"))
            #        #assert len(p) == 1, "every dir can only have one xyz file."
            #        xyz_path = sp / args.name
            #        if xyz_path.exists():
            #            structure_files.append(xyz_path)
            #structure_files.sort()
            pass

        return structure_files
    
    def run(self, structure_source, index_text=None, *args, **kwargs):
        """"""
        self.set_environs()
        
        indices = convert_indices(index_text, index_convention="py")

        structure_files = self._parse_structures(structure_source)
        for stru_file in structure_files:
            # each single file should have different output directory
            self._irun(stru_file, indices)

        return
    
    def _irun(self, stru_file, indices: List[int]=None, *args, **kwargs):
        """"""
        # read structures
        #frames = read(stru_file, indices)
        #start, end = indices.split(":")
        #if start == "":
        #    start = 0
        #else:
        #    start = int(start)
        #print("{} structures in {} from {}\n".format(len(frames), stru_file, start))
        #print(stru_file)
        frames = read(stru_file, ":")
        nframes = len(frames)
        #print(frames)
        #print(indices)

        working_directory = Path.cwd()

        if indices:
            frames = [frames[i] for i in indices]
        else:
            indices = range(nframes)

        # init paths
        out_xyz = working_directory / "calculated_0.xyz"

        with open(out_xyz, "w") as fopen:
            fopen.write("")

        # - run calc
        print("\n===== Calculation Stage =====\n")

        for idx, atoms in zip(indices, frames):
            step = idx
            print("\n\nStructure Number %d\n" %step)
            dpath = working_directory / f"vasp_0_{step}"

            if not dpath.exists():
                self.calc.reset()
                self.calc.directory = dpath
                atoms.calc = self.calc

                with CustomTimer(name="vasp-calculation"):
                    _ = atoms.get_forces()
                
            if self.calc.read_convergence():
                new_atoms = self._read_single_results(dpath)[-1]
                new_atoms.info["step"] = step
                print("final energy: ", new_atoms.get_potential_energy())

                # check forces
                #maxforce = np.max(np.fabs(new_atoms.get_forces(apply_constraint=True)))
                #print(new_atoms.get_forces())
                #print("maxforce: ", maxforce)
                #print("fmax: ", vasp_machine.fmax)
                #if not (maxforce < np.fabs(self.vasp_machine.fmax)): # fmax is ediffg so is negative
                #    # TODO: recalc structure
                #    #raise RuntimeError(f"{dpath} structure is not finished...")
                #    print(f"{dpath} structure is not finished...")

                # save structure
                write(out_xyz, new_atoms, append=True)
            else:
                print(f"{dpath.name} did not converge.")

        return
    
    def _read_results(self, main_dir):
        """"""
        # TODO: replace this with an object
        main_dir = Path(main_dir)
        print("main_dir: ", main_dir)
        # - parse subdirs
        #vasp_main_dirs = []
        #for p in main_dir.iterdir():
        #    calc_file = p / self.frames_name
        #    if p.is_dir() and calc_file.exists():
        #        vasp_main_dirs.append(p)
        #print(vasp_main_dirs) 
        # - parse vasp dirs
        vasp_dirs = [] # NOTE: converged vasp dirs
        for p in main_dir.glob(self.vasp_dir_pattern):
            if p.is_dir():
                self.calc.directory = str(p)
                if self.calc.read_convergence():
                    vasp_dirs.append(p)

        # - read trajectories
        traj_frames = []
        if len(vasp_dirs) > 0:
            traj_bundles = Parallel(n_jobs=config.NJOBS)(
                delayed(self._read_single_results)(p, indices=None) for p in vasp_dirs
            ) # TODO: indices

            for frames in traj_bundles:
                traj_frames.extend(frames)

        return traj_frames
    
    def _read_single_results(self, vasp_dir, indices=None, verbose=False):
        """ read results from single vasp dir
        """
        # get vasp calc params to add extra_info for atoms
        #fmax = self.fmax # NOTE: EDIFFG negative for forces?

        # read info
        # TODO: check electronic convergence
        #vasprun = pstru / "vasprun.xml"
        #if vasprun.exists() and vasprun.stat().st_size > 0:
        #    pass
        vasprun = vasp_dir / "vasprun.xml"

        # - read structures
        if indices is None:
            traj_frames = read(vasprun, ":")
            energies = [a.get_potential_energy() for a in traj_frames]
            maxforces = [np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in traj_frames] # TODO: applt cons?

            #print(f"--- vasp info @ {pstru.name} ---")
            #print("nframes: ", len(traj_frames))
            #print("last energy: ", energies[-1])
            #print("last maxforce: ", maxforces[-1])
            #print("force convergence: ", fmax)

            last_atoms = traj_frames[-1]
            last_atoms.info["source"] = vasp_dir.name
            last_atoms.info["maxforce"] = maxforces[-1]
            #if maxforces[-1] <= fmax:
            #    write(pstru / (pstru.name + "_opt.xsd"), last_atoms)
            #    print("write converged structure...")
            last_atoms = [last_atoms]
        else:
            #print(f"--- vasp info @ {pstru.name} ---")
            last_atoms = read(vasprun, indices)
            #print("nframes: ", len(last_atoms))
            #nconverged = 0
            #for a in last_atoms:
            #    maxforce = np.max(np.fabs(a.get_forces(apply_constraint=True)))
            #    if maxforce < fmax:
            #        a.info["converged"] = True
            #        nconverged += 1
            #print("nconverged: ", nconverged)

        return last_atoms


if __name__ == "__main__":
    pass