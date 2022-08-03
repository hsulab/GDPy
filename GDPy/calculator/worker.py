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

from joblib import Parallel, delayed

from ase.io import read, write
from ase.calculators.vasp import Vasp

from GDPy import config
from GDPy.utils.command import parse_input_file, convert_indices
from GDPy.calculator.vasp import VaspMachine

class VaspMachine2():

    frames_name = "calculated_0.xyz"
    vasp_dir_pattern = "vasp_*"

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
        print(self.calc.asdict())

        return
    
    def _prepare_calculation(self):
        """ deal with many calculations in one folder
        """
        # -- update params with systemwise info
        # TODO: dump vasp ase-calculator to json
        calc_params = calc_dict.copy()
        for k in calc_params.keys():
            if calc_params[k] == "system":
                calc_params[k] = self.init_systems[slabel][k]
        # -- create params file
        with open(sorted_fp_path/"vasp_params.json", "w") as fopen:
            json.dump(calc_params, fopen, indent=4)
        # -- create job script
        machine_params = machine_dict.copy()
        machine_params["job-name"] = slabel+"-fp"

        # TODO: mpirun or mpiexec, move this part to machine object
        command = calc_params["command"]
        if command.strip().startswith("mpirun") or command.strip().startswith("mpiexec"):
            ntasks = command.split()[2]
        else:
            ntasks = 1
        # TODO: number of nodes?
        machine_params.update(**{"nodes": "1-1", "ntasks": ntasks, "cpus-per-task": 1, "mem-per-cpu": "4G"})

        machine = SlurmMachine(**machine_params)
        #"gdp vasp work ../C3O3Pt36.xyz -in ../vasp_params.json"
        machine.user_commands = "gdp vasp work {} -in {}".format(
            str(final_selected_path.resolve()), (sorted_fp_path/"vasp_params.json").resolve()
        )
        machine.write(sorted_fp_path/"vasp.slurm")
        # -- submit?

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
        
        #print(vasp_dirs)
        
        # - read trajectories
        traj_bundles = Parallel(n_jobs=config.NJOBS)(
            delayed(self._read_single_results)(p, indices=None) for p in vasp_dirs
        ) # TODO: indices

        traj_frames = []
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