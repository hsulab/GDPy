#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" A worker that manages a series of dynamics tasks
    worker needs a machine to dertermine whether run by serial
    or on cluster
"""

import time
import subprocess

from pathlib import Path
from typing import Union, List

import numpy as np

from ase.io import read, write

from GDPy.utils.command import parse_input_file, convert_indices
from GDPy.calculator.vasp import VaspMachine


class VaspWorker():

    """ run many dynamics tasks from different initial configurations
    """

    def __init__(self, vasp_params: dict, single_task_command: str, directory=Path.cwd(), *args, **kwargs):
        """ vasp_params :: vasp calculation related parameters
        """
        self.directory = directory
        self.single_task_command = single_task_command

        # parse vasp parameters

        # create a vasp machine
        self.vasp_machine = VaspMachine(**vasp_params)

        return
    
    def _parse_structures(self, structure_source: Union[str,Path]):
        """"""
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

        if indices:
            frames = [frames[i] for i in indices]
        else:
            indices = range(nframes)

        # init paths
        out_xyz = self.directory / "calculated_0.xyz"

        with open(out_xyz, "w") as fopen:
            fopen.write("")

        # run calc
        print("\n===== Calculation Stage =====\n")
        #print(indices)
        #print(frames)
        print(self.single_task_command)
        for idx, atoms in zip(indices, frames):
            step = idx
            print("\n\nStructure Number %d\n" %step)
            dpath = self.directory / f"vasp_0_{step}"

            if not dpath.exists():
                self.vasp_machine.create(atoms, dpath)

                # ===== run command =====
                st = time.time()
                proc = subprocess.Popen(
                    self.single_task_command, shell=True, cwd=dpath, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding = "utf-8"
                )
                errorcode = proc.wait()
                msg = "Message: " + "".join(proc.stdout.readlines())
                print(msg)
                if errorcode:
                    raise ValueError("Error at %s." %(dpath))
                et = time.time()
                print("calc time: ", et-st)

                # read optimised
                new_atoms = self.vasp_machine.get_results(dpath)
                new_atoms.info["step"] = step
                print("final energy: ", new_atoms.get_potential_energy())
            else:
                new_atoms = self.vasp_machine.get_results(dpath)
                # check forces
                maxforce = np.max(np.fabs(new_atoms.get_forces(apply_constraint=True)))
                #print(new_atoms.get_forces())
                #print("maxforce: ", maxforce)
                #print("fmax: ", vasp_machine.fmax)
                if not (maxforce < np.fabs(self.vasp_machine.fmax)): # fmax is ediffg so is negative
                    # TODO: recalc structure
                    #raise RuntimeError(f"{dpath} structure is not finished...")
                    print(f"{dpath} structure is not finished...")

            # save structure
            write(out_xyz, new_atoms, append=True)

        return


if __name__ == "__main__":
    pass