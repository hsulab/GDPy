#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import copy
import json
import argparse
import subprocess 
import warnings
from typing import Union, List, NoReturn
from collections import Counter

import shutil

from pathlib import Path

import numpy as np 

from ase import Atoms 
from ase.io import read, write
from ase.constraints import FixAtoms

from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.builder.constraints import parse_constraint_info
from GDPy.utils.command import run_command, parse_input_file
from GDPy.computation.utils import create_single_point_calculator
from GDPy.computation.driver import AbstractDriver

""" wrap ase-vasp into a few utilities
"""

def collect_vasp():
    cwd = Path.cwd()
    for p in cwd.glob("*"):
        vasprun = p / "vasprun.xml"
        if vasprun.exists():
            print("==== {} =====".format(p))
            frames = read(vasprun, ":")
            atoms = frames[-1]
            forces = atoms.get_forces()
            maxforce = np.max(np.fabs(forces))
            if maxforce < 0.05:
                print("energy: ", atoms.get_potential_energy())
                write(p.name + "_opt.xsd", atoms)
                write(p.name + "_optraj.xtd", frames)
                write(p.name + "_optraj.xyz", frames)
            else:
                print("not converged: ")
                write(p.name + "_failed.xtd", frames)

    return


# vasp utils
def read_sort(directory):
    """Create the sorting and resorting list from ase-sort.dat.
    If the ase-sort.dat file does not exist, the sorting is redone.
    """
    sortfile = directory / 'ase-sort.dat'
    if os.path.isfile(sortfile):
        sort = []
        resort = []
        with open(sortfile, 'r') as fd:
            for line in fd:
                s, rs = line.split()
                sort.append(int(s))
                resort.append(int(rs))
    else:
        # warnings.warn(UserWarning, 'no ase-sort.dat')
        raise ValueError('no ase-sort.dat')

    return sort, resort

class VaspDriver(AbstractDriver):

    name = "vasp"

    # - defaults
    default_task = "min"
    supported_tasks = ["min", "md"]

    default_init_params = {
        "min": {
            "min_style": 1, # ibrion
            #"min_modify": "integrator verlet tmax 4"
        },
        "md": {
            "md_style": "nvt",
            "velocity_seed": 1112,
            "timestep": 1.0, # fs
            "temp": 300, # K
            "Tdamp": 100, # fs
            "pres": 1.0, # atm
            "Pdamp": 100
        }
    }

    default_run_params = {
        "min": {
            "ediffg": 0.05,
            "nsw": 0,
        },
        "md": {
            "nsw": 0
        }
    }

    param_mapping = {
        "min_style": "ibrion",
        "fmax": "ediffg",
        "steps": "nsw"
    }

    # - system depandant params
    syswise_keys = [
       "system", "kpts"
    ]

    def _parse_params(self, params_: dict) -> NoReturn:
        super()._parse_params(params_)

        # - update task
        # self.init_params.update(task=self.task)

        # - special settings
        self.run_params = self.__set_special_params(self.run_params)

        return 
    
    def __set_special_params(self, params):
        """"""
        params["ediffg"] *= -1.

        return params
    
    def run(self, atoms_, read_exists=True, extra_info=None, *args, **kwargs):
        """"""
        atoms = atoms_.copy()

        # - backup old params
        # TODO: change to context message?
        calc_old = atoms.calc 
        params_old = copy.deepcopy(self.calc.parameters)

        # - set special keywords
        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)

        # - run params
        kwargs = self._map_params(kwargs)

        run_params = self.run_params.copy()
        run_params.update(kwargs)

        # - init params
        run_params.update(**self.init_params)

        run_params = self.__set_special_params(run_params)

        cons_text = run_params.pop("constraint", None)
        mobile_indices, frozen_indices = parse_constraint_info(atoms, cons_text, ret_text=False)
        if frozen_indices:
            atoms._del_constraints()
            atoms.set_constraint(FixAtoms(indices=frozen_indices))
        
        run_params["system"] = self.directory.name

        self.calc.set(**run_params)

        # BUG: ase 3.22.1 no special params in param_state
        #calc_params["inputs"]["lreal"] = self.calc.special_params["lreal"] 

        atoms.calc = self.calc
        print(atoms)

        # - run dynamics
        try:
            # NOTE: some calculation can overwrite existed data
            if read_exists:
                converged = False
                if (self.directory/"OUTCAR").exists():
                    converged = atoms.calc.read_convergence()
                if not converged:
                    _ = atoms.get_forces()
            else:
                _  = atoms.get_forces()
                converged = atoms.calc.read_convergence()
        except OSError:
            converged = False
        
        # NOTE: always use dynamics calc
        # TODO: should change positions and other properties for input atoms?
        traj_frames = self.read_trajectory()
        new_atoms = traj_frames[-1]

        if extra_info is not None:
            new_atoms.info.update(**extra_info)

        # - reset params
        self.calc.parameters = params_old
        self.calc.reset()
        if calc_old is not None:
            atoms.calc = calc_old

        return new_atoms
    
    def read_trajectory(self, *args, **kwargs) -> List[Atoms]:
        """"""
        vasprun = self.directory / "vasprun.xml"

        # - read structures
        try:
            traj_frames_ = read(vasprun, ":")
            traj_frames = []

            # - sort frames
            sort, resort = read_sort(self.directory)
            for sorted_atoms in traj_frames_:
                input_atoms = create_single_point_calculator(sorted_atoms, resort, "vasp")
                traj_frames.append(input_atoms)
        except:
            atoms = Atoms()
            atoms.info["error"] = str(self.directory)
            traj_frames = [atoms]

        return traj_frames


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

    script_name = "vasp.slurm"

    def __init__(
        self, 
        incar, 
        pp_path, 
        vdw_path=None, 
        vasp_script=None, # job script
        isinteractive = False, # if output input info to screen
        **kwargs # parse kpts, constraints
    ):
        """"""
        self.__incar = incar

        self.vasp_script = None
        if vasp_script is not None:
            self.vasp_script = SlurmMachine(use_gpu=False) # TODO: change to general machine
            self.vasp_script.update(vasp_script)

        self.pp_path = pp_path
        self.vdw_path = vdw_path

        self.isinteractive = isinteractive

        # kpoints mesh
        self.__kpts = kwargs.get("kpts", None)
        self.__constraint = kwargs.get("constraint", None)

        self.__set_environs()

        # TODO: electronic convergence
        # geometric convergence
        self.fmax = 0.05

        return
    
    def __set_environs(self):
        # ===== environs TODO: from inputs 
        # pseudo 
        if "VASP_PP_PATH" in os.environ.keys():
            os.environ.pop("VASP_PP_PATH")
        os.environ["VASP_PP_PATH"] = self.pp_path

        # vdw 
        vdw_envname = "ASE_VASP_VDW"
        if vdw_envname in os.environ.keys():
            _ = os.environ.pop(vdw_envname)
        os.environ[vdw_envname] = self.vdw_path

        return
    
    def _init_creator(self):
        """ NOTE: this can be used to access 
        """
        # ===== set basic params
        vasp_creator = GenerateVaspInput()
        if self.__incar is not None:
            vasp_creator.set_xc_params("PBE") # NOTE: since incar may not set GGA
            vasp_creator.read_incar(self.__incar)
        else:
            vasp_creator.set(**self.default_parameters)
        
        # geometric convergence
        self.fmax = np.fabs(vasp_creator.exp_params.get("ediffg", 0.05))

        return vasp_creator

    def create(
        self, 
        atoms, 
        directory=Path("vasp-test"),
        task = "opt"
    ):
        """ create a single calculation directory
        """
        # - creator
        vasp_creator = self._init_creator()

        # geometric convergence
        self.fmax = np.fabs(vasp_creator.exp_params.get("ediffg", 0.05))

        # overwrite some structure specific parameters
        vasp_creator.set(system=directory.name) # SYSTEM

        # different tasks
        if task == "opt":
            pass
        elif task == "copt":
            pass
        elif task == "freq":
            vasp_creator.set(nsw=1)
            vasp_creator.set(ibrion=5)
            vasp_creator.set(nfree=2)
            vasp_creator.set(potim=0.015)

        # TODO: use not gamma-centred mesh?
        vasp_creator.set(gamma=True)
        if self.__kpts is None:
            # default equals kspacing = 20
            kpts = np.linalg.norm(atoms.cell, axis=1).tolist()
            kpts = [int(20./k)+1 for k in kpts] 
        else:
            # NOTE: this vasp setting maybe used for various systems
            # this self.__kpts should be None if not set in the first place
            kpts = self.__kpts
        vasp_creator.set(kpts=kpts)

        # set constraint
        if self.__constraint is not None:
            # use lammps convertion
            cons_indices = []
            for s in self.__constraint.strip().split():
                start, end = s.split(":")
                cons_indices.extend(list(range(int(start)-1,int(end))))
            cons = FixAtoms(indices=cons_indices)
            atoms.set_constraint(cons)

        # write inputs
        if not directory.exists():
            directory.mkdir()
        else:
            print(f"overwrite previous input files in {directory}...")

        vasp_creator.initialize(atoms)
        vasp_creator.write_input(atoms, directory)

        if self.vasp_script is not None:
            # vasp_script = self.vasp_script.copy() # TODO: add copy
            self.vasp_script.machine_dict["job-name"] = directory.name
            self.vasp_script.write(directory / self.script_name)
        
        # output info
        if self.isinteractive:
            # --- summary ---
            content = '\n>>>>> Modified ASE for VASP <<<<<\n'
            content += '    directory -> %s\n' %directory 
            print(content)

            # --- poscar ---
            symbols = atoms.get_chemical_symbols() 
            all_atoms = Counter(symbols) 
            cons = atoms.constraints
            # print(cons)
            if len(cons) == 1:
                cons_indices = cons[0].get_indices() 
                print("cons_indices: ", cons_indices)
                fixed_symbols = [symbols[i] for i in cons_indices]
                fixed_atoms = Counter(fixed_symbols)
            else:
                fixed_atoms = all_atoms.copy()
                for key in fixed_atoms.keys():
                    fixed_atoms[key] = 0

            natoms = len(atoms) 
            nfixed = np.sum(list(fixed_atoms.values()))

            content = "\nPOSCAR -->\n"
            content += "    \033[4m Element       Numbers      \033[0m\n"
            for sym in all_atoms.keys():
                content += "         %2s \033[1;33m%4d\033[0m \033[32m%4d\033[0m(T)\033[31m%4d\033[0m(F)\n"\
                        %(sym, all_atoms[sym], all_atoms[sym]-fixed_atoms[sym], fixed_atoms[sym])
            content += "    \033[4m                            \033[0m\n"
            content += "      %2s \033[1;33m%4d\033[0m \033[32m%4d\033[0m(T)\033[31m%4d\033[0m(F)\n"\
                        %('Total', natoms, natoms-nfixed, nfixed)
            print(content) 

            # --- kpoints ---
            content = "KPOINTS -->\n"
            content += "     Set k-point -> \033[1;35m{:s}\033[0m\n".format(str(kpts))
            print(content)

            # --- copt --- 
            # TODO: move this to task part
            copt = atoms.info.get("copt", None)
            if copt: 
                # If constrained get distance and create fort.188
                symbols = atoms.get_chemical_symbols() 
                ca, cb = atoms.info["copt"][0], atoms.info["copt"][1] 
                pt1, pt2 = atoms.positions[ca], atoms.positions[cb] 

                # Use Ax=b convert to cartisan coordinate
                distance = np.linalg.norm(pt1 - pt2) # NOTE: should consider MIC

                # Create fort.188
                ts_content = "1\n3\n6\n4\n0.04\n%-5d%-5d%f\n0\n" % \
                    (ca+1, cb+1, distance)

                with open(os.path.join(directory, "fort.188"), "w") as fopen:
                    fopen.write(ts_content)
    
                vasp_creator.set(ibrion=1)

                content += "\n"
                content += "     fort.188 has been created.\n"
                content += "     " + "-"*20 + "\n"
                content += "     atom number: {:<5d}{:<5d}\n".format(ca+1, cb+1)
                content += "     atom name: {} {}\n".format(symbols[ca], symbols[cb])
                content += "     distance: {:f}\n".format(distance)
                content += "     " + "-"*20 + "\n"

                # Set IBRION = 1
                content += "     Note: IBRION has been set to 1.\n"
                print(content)

            # --- potcar ---

        return
    
    def submit(self, directory):
        """"""
        run_command(
            directory, "{} {}".format(self.vasp_script.SUBMIT_COMMAND, self.script_name),
            comment="submit a vasp calculation"
        )

        return
    
    def get_results(
        self,
        vasp_dir, # directory where vasp was performed
        fmax: float = 0.05, # geometric convergence
        extra_info: dict = None # info to add in atoms
    ) -> Union[None, Atoms]:

        atoms = None

        print(vasp_dir)
        vasp_dir = Path(vasp_dir)
        # TODO: this only works for vasp, should be more general
        vasprun = vasp_dir / "vasprun.xml"
        if vasprun.exists() and vasprun.stat().st_size > 0:
            frames = read(vasprun, ':')
            print("nframes: ", len(frames))
            atoms_sorted = frames[-1] # TODO: for now, only last one of interest
            # resort
            sort, resort = read_sort(vasp_dir)
            atoms = create_single_point_calculator(atoms_sorted, resort, "vasp")

            # add extra info
            if extra_info is not None:
                atoms.info.update(extra_info)
        else:
            print("the job isnot running...")

        return atoms

    def read_results(self, pstru, indices=None, custom_incar=None, verbose=False):
        """"""
        # get vasp calc params to add extra_info for atoms
        fmax = self.fmax # NOTE: EDIFFG negative for forces?

        # read info
        # TODO: check electronic convergence
        vasprun = pstru / "vasprun.xml"
        if vasprun.exists() and vasprun.stat().st_size > 0:
            pass

        # - read structures
        if indices is None:
            traj_frames = read(vasprun, ":")
            energies = [a.get_potential_energy() for a in traj_frames]
            maxforces = [np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in traj_frames] # TODO: applt cons?

            print(f"--- vasp info @ {pstru.name} ---")
            print("nframes: ", len(traj_frames))
            print("last energy: ", energies[-1])
            print("last maxforce: ", maxforces[-1])
            print("force convergence: ", fmax)

            last_atoms = traj_frames[-1]
            last_atoms.info["source"] = pstru.name
            last_atoms.info["maxforce"] = maxforces[-1]
            if maxforces[-1] <= fmax:
                write(pstru / (pstru.name + "_opt.xsd"), last_atoms)
                print("write converged structure...")
        else:
            print(f"--- vasp info @ {pstru.name} ---")
            last_atoms = read(vasprun, indices)
            print("nframes: ", len(last_atoms))
            nconverged = 0
            for a in last_atoms:
                maxforce = np.max(np.fabs(a.get_forces(apply_constraint=True)))
                if maxforce < fmax:
                    a.info["converged"] = True
                    nconverged += 1
            print("nconverged: ", nconverged)

        return last_atoms
    
    def __icalculate(self):
        # TODO: move vasp calculation to this one

        return

class VaspQueue:

    prefix = "cand"

    def __init__(
            self, data_connection, tmp_folder, 
            vasp_machine,
            n_simul, prefix,
            repeat = -1,
            submit_command="sbatch", stat_command="squeue"
        ):
        self.dc = data_connection
        self.n_simul = n_simul

        self.vasp_machine = vasp_machine # deal with vasp inputs and outputs

        self.prefix = prefix

        self.repeat = repeat

        self.submit_command = submit_command
        self.stat_command = stat_command
        self.calc_command = "{0} {1}".format(self.submit_command, self.vasp_machine.script_name) 

        self.tmp_folder = Path(tmp_folder)

        self.__cleanup__()

        return

    def relax(self, a: Atoms):
        """ Add a structure to the queue. This method does not fail
            if sufficient jobs are already running, but simply
            submits the job. """
        # check candidate
        self.__cleanup__()
        self.dc.mark_as_queued(a) # this marks relaxation is in the queue
        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)
        
        # create atom structure
        dpath = Path("{0}/{1}{2}".format(self.tmp_folder, self.prefix, a.info["confid"]))
        self.vasp_machine.create(a, dpath)

        # submit job
        msg = run_command(dpath, self.calc_command, comment="submitting job", timeout=30)

        return

    def enough_jobs_running(self):
        """ Determines if sufficient jobs are running. """
        return self.number_of_jobs_running() >= self.n_simul

    def number_of_jobs_running(self):
        return len(self.__get_running_job_ids())

    def __get_running_job_ids(self):
        """ Determines how many jobs are running. The user
            should use this or the enough_jobs_running method
            to verify that a job needs to be started before
            calling the relax method."""
        # self.__cleanup__()
        # TODO: reform
        p = subprocess.Popen(
            ['`which {0}` -u `whoami` --format=\"%.12i %.12P %.24j %.4t %.12M %.12L %.5D %.4C\"'.format(self.stat_command)],
            shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            close_fds=True, universal_newlines=True
        )
        fout = p.stdout
        lines = fout.readlines()
        # print(''.join(lines)) # output of squeue
        confids = []
        for line in lines[1:]: # skipe first info line
            data = line.strip().split()
            jobid, name, status = data[0], data[2], data[3]
            if name.startswith(self.prefix) and status in ['R','Q','PD']: # TODO: move status to machine itself
                #print(jobid)
                #print(name)
                indices = re.match(self.prefix+"*", name).span()
                if indices is not None:
                    confid = int(name[indices[1]:])
                confids.append(int(confid))
        #print(len(confids))

        return confids
    
    def __get_failed_job_ids(self):
        """"""
        fpath = Path(self.tmp_folder / "FAILED_IDS")
        if fpath.exists():
            failed_ids = np.loadtxt(fpath, dtype=int).tolist()
        else:
            failed_ids = []

        return failed_ids
    
    def check_status(self) -> List[Atoms]:
        """
        check job status
        return a list of candidates for database
        """
        candidates = []

        print('\n===== check job status =====')
        running_ids = self.__get_running_job_ids()
        failed_ids = self.__get_failed_job_ids()

        confs = self.dc.get_all_candidates_in_queue() # conf ids
        print("cand in queue: ", confs)
        for confid in confs:
            # BUG: calculation may be still runniing. Check if finished before resubmit
            if confid in running_ids:
                print("\n --- current cand: ", cand_dir, "is running...")
                continue
            cand_dir = self.tmp_folder / (self.prefix+str(confid))
            print("\n --- current cand: ", cand_dir)
            # TODO: this only works for vasp, should be more general
            atoms = self.vasp_machine.get_results(cand_dir)

            # check forces
            if isinstance(atoms, Atoms):
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                maxforce = np.max(np.fabs(forces))
                print("energy: {:.4f}  maxforce: {:.4f}".format(energy, maxforce))
                if maxforce <= self.vasp_machine.fmax: # check geometric convergence
                    print("is converged")
                    atoms.info["confid"] = confid
                    # add few information
                    atoms.info["data"] = {}
                    atoms.info["key_value_pairs"] = {"extinct": 0}
                    candidates.append(atoms)
                else:
                    # still leave queued=1 in the db, but resubmit
                    print("not converged...")
                    cur_repeat = 1
                    fpath = cand_dir / "repeat.ga"
                    if fpath.exists():
                        cur_repeat = np.loadtxt(fpath, dtype=int)
                        print("cur_repeat: ", cur_repeat)
                    if cur_repeat < self.repeat and confid not in failed_ids:
                        # NOTE: read bak.x.vasprun.xml of relaxation trajectory for MLP
                        print("backup old data...")
                        saved_cards = ["OUTCAR", "vasprun.xml"]
                        for card in saved_cards:
                            card_path = cand_dir / card
                            bak_fmt = ("bak.{:d}."+card)
                            idx = 0
                            while True:
                                bak_card = bak_fmt.format(idx)
                                if not Path(bak_card).exists():
                                    saved_card_path = cand_dir / bak_card
                                    shutil.copy(card_path, saved_card_path)
                                    break
                                else:
                                    idx += 1
                        # update positions
                        shutil.copy(cand_dir / "CONTCAR", cand_dir / "POSCAR")
                        print("resubmit job...")
                        msg = run_command(cand_dir, self.calc_command, comment="submitting job", timeout=30)
                        cur_repeat += 1
                        np.savetxt(fpath, np.array([cur_repeat]), fmt="%d")
                    else:
                        print("save failed configuration to the database")
                        atoms.info["confid"] = confid
                        # add few information
                        atoms.info["data"] = {}
                        atoms.info["key_value_pairs"] = {"extinct": 0, "failed": 1}
                        candidates.append(atoms)

        return candidates
    
    def __cleanup__(self):
        """
        """
        # self.check_status()

        return

def run_calculation(work_path, stru_file, indices, vasp_machine, command):
    """"""
    # read structures
    frames = read(stru_file, indices)
    start, end = indices.split(":")
    if start == "":
        start = 0
    else:
        start = int(start)
    print("{} structures in {} from {}\n".format(len(frames), stru_file, start))

    # init paths
    out_xyz = work_path / "calculated_0.xyz"

    with open(out_xyz, "w") as fopen:
        fopen.write("")

    # run calc
    print("\n===== Calculation Stage =====\n")
    for idx, atoms in enumerate(frames):
        step = idx+start
        print("\n\nStructure Number %d\n" %step)
        dpath = work_path / f"./vasp_0_{step}"

        if not dpath.exists():
            vasp_machine.create(atoms, dpath)

            # ===== run command =====
            st = time.time()
            proc = subprocess.Popen(
                command, shell=True, cwd=dpath, 
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
            new_atoms = vasp_machine.get_results(dpath)
            new_atoms.info["step"] = step
            print("final energy: ", new_atoms.get_potential_energy())
        else:
            new_atoms = vasp_machine.get_results(dpath)
            # check forces
            maxforce = np.max(np.fabs(new_atoms.get_forces(apply_constraint=True)))
            #print(new_atoms.get_forces())
            #print("maxforce: ", maxforce)
            #print("fmax: ", vasp_machine.fmax)
            if not (maxforce < np.fabs(vasp_machine.fmax)): # fmax is ediffg so is negative
                # TODO: recalc structure
                #raise RuntimeError(f"{dpath} structure is not finished...")
                print(f"{dpath} structure is not finished...")

        # save structure
        write(out_xyz, new_atoms, append=True)

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
    parser.add_argument(
        "-d", "--dirs", nargs="*", default=[],
        help="dirs"
    )
    parser.add_argument(
        "-n", "--name", default="miaow-sel.xyz", 
        help="xyz file"
    )

    args = parser.parse_args()

    # parse vasp parameters
    # parse vasp parameters
    input_dict = parse_input_file(args.parameters)
    input_dict["isinteractive"] = True
    print(input_dict)

    # create a vasp machine
    vasp_machine = VaspMachine(**input_dict)

    stru_path = Path(args.STRUCTURES)
    if stru_path.is_file():
        # run calculation 
        run_calculation(Path.cwd(), args.STRUCTURES, args.indices, vasp_machine, input_dict["command"])
    else:
        # all structures share same indices and vasp_machine
        structure_files = []
        for sp in stru_path.iterdir():
            if sp.is_dir() and (not args.dirs or sp.name in args.dirs):
                #p = list(sp.glob("*.xyz"))
                #assert len(p) == 1, "every dir can only have one xyz file."
                xyz_path = sp / args.name
                if xyz_path.exists():
                    structure_files.append(xyz_path)
        structure_files.sort()
        print(structure_files)

        print("run structures...")
        for p in structure_files:
            print(p)
            run_calculation(p.parent, p, args.indices, vasp_machine, input_dict["command"])
