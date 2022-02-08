#!/usr/bin/env python3

import os
import time
import json
import argparse
import subprocess 

from pathlib import Path 

import numpy as np 

from ase import Atoms 
from ase.io import read, write
from ase.calculators.vasp.create_input import GenerateVaspInput

""" wrap ase-vasp into a few utilities
"""


class VaspMachine():

    default_parameters = {
        # ase specific
        "xc": "pbe",
        "gamma": True,
        "kpts": (1,1,1),
        # output
        "nwrite": 2, 
        "istart": 0, 
        "lcharg": False, 
        "lwave": False, 
        "lorbit": 10,
        # parallel
        "npar": 4,
        # electronic
        "encut": 400,
        "prec": "Normal",
        "ediff": 1E-5,
        "nelm": 180, 
        "nelmin": 6, 
        "ispin": 1,
        "ismear": 1,
        "sigma": 0.2,
        "algo": "Fast", 
        "lreal": "Auto", 
        "isym": 0, 
        # geometric
        "ediffg": -0.05,
        "nsw": 200,
        "ibrion": 2,
        "isif": 2,
        "potim": 0.2, 
    } 

    def __init__(self, command, incar, pp_path, vdw_path=None, **kwargs):
        """"""
        self.command = command
        self.__incar = incar

        self.pp_path = pp_path
        self.vdw_path = vdw_path

        self.__set_environs()

        return
    
    def __set_environs(self):
        # ===== environs TODO: from inputs 
        # pseudo 
        if 'VASP_PP_PATH' in os.environ.keys():
            os.environ.pop('VASP_PP_PATH')
        os.environ['VASP_PP_PATH'] = self.pp_path

        # vdw 
        vdw_envname = 'ASE_VASP_VDW'
        if vdw_envname in os.environ.keys():
            _ = os.environ.pop(vdw_envname)
        os.environ[vdw_envname] = self.vdw_path

        return

    def create_by_ase(
        self, 
        atoms, 
        kpts = None,
        directory=Path('vasp-test')
    ):
        # ===== set basic params
        vasp_creator = GenerateVaspInput()
        if self.__incar is not None:
            vasp_creator.set_xc_params("PBE") # NOTE: since incar may not set GGA
            vasp_creator.read_incar(self.__incar)
        else:
            vasp_creator.set(**self.default_parameters)

        # overwrite some structure specific parameters
        vasp_creator.set(system=directory.name)

        # TODO: use not gamma-centred mesh?
        vasp_creator.set(gamma=True)
        if kpts is None:
            kpts = np.linalg.norm(atoms.cell, axis=1).tolist()
            kpts = [int(20./k)+1 for k in kpts] 
        vasp_creator.set(kpts=kpts)

        # write inputs
        if not directory.exists():
            directory.mkdir()
        else:
            print(f"overwrite previous input files in {directory}...")

        vasp_creator.initialize(atoms)
        vasp_creator.write_input(atoms, directory)

        return

def run_calculation(stru_file, indices, vasp_machine, kpts):
    """"""
    # read structures
    frames = read(stru_file, indices)
    start, end = indices.split(":")
    if start == "":
        start = 0
    else:
        start = int(start)
    print("{} structures in {} from {}\n".format(len(frames), stru_file, start))

    # run calc
    print("\n===== Calculation Stage =====\n")
    for idx, atoms in enumerate(frames):
        step = idx+start
        print("\n\nStructure Number %d\n" %step)
        dpath = Path(f"./vasp_0_{step}")
        vasp_machine.create_by_ase(atoms, kpts, dpath)
        # run command
        st = time.time()
        proc = subprocess.Popen(
            vasp_machine.command, shell=True, cwd=dpath, 
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
        vasprun = dpath / "vasprun.xml"
        frames = read(vasprun, ":")
        print("number of frames: ", len(frames))
        new_atoms = frames[-1]
        new_atoms.info["step"] = step
        print("final energy: ", new_atoms.get_potential_energy())

        # save structure
        write("calculated_0.xyz", new_atoms, append=True)

    return


if __name__ == "__main__": 
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "STRUCTURES", 
        help="input structures stored in xyz format file"
    )
    parser.add_argument(
        "-p", "--parameters", default="calc_vasp.json", 
        help="calculator-related parameters in json format file"
    )
    parser.add_argument(
        "-i", "--indices", default=":", 
        help="frame selection e.g. 0:100 that calculates structure 1 to 100"
    )

    args = parser.parse_args()

    # parse vasp parameters
    with open(args.parameters, "r") as fopen:
        input_dict = json.load(fopen)
    print(input_dict)

    vasp_machine = VaspMachine(**input_dict)

    # run calculation 
    run_calculation(args.STRUCTURES, args.indices, vasp_machine, input_dict.get("kpts", None))
