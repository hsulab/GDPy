#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
import numpy as np

import copy

from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError

from GDPy.calculator.dynamics import AbstractDynamics

from ase.calculators.singlepoint import SinglePointCalculator

"""
output files by lasp NVE-MD
    allfor.arc  allkeys.log  allstr.arc  firststep.restart
    md.arc  md.restart  vel.arc
"""


class LaspDynamics(AbstractDynamics):

    """ local optimisation
    """

    def run(self, atoms, **kwargs):

        calc_old = atoms.calc
        params_old = copy.deepcopy(self.calc.parameters)

        # set special keywords
        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)

        kwargs["explore_type"] = "ssw" # must be this
        kwargs["SSW.SSWsteps"] = 1 # local optimisation

        kwargs["SSW.ftol"] = kwargs.pop("fmax", 0.05)
        kwargs["SSW.MaxOptstep"] = kwargs.pop("steps", 300)

        self.calc.set(**kwargs)
        atoms.calc = self.calc

        # change dir this may be done in calc
        # if not self._directory_path.exists():
        #     self._directory_path.mkdir(parents=True)

        # run dynamics
        try:
            _  = atoms.get_forces()
        except OSError:
            converged = False
        else:
            converged = True
        
        # read new atoms positions
        new_atoms = read(
            self._directory_path / "allstr.arc", "-1", format="dmol-arc"
        )
        sp_calc = SinglePointCalculator(new_atoms, **copy.deepcopy(self.calc.results))
        new_atoms.calc = sp_calc

        # restore to old calculator
        # self.calc.parameters = params_old
        # self.calc.reset() # TODO: should use this?
        # if calc_old is not None:
        #     atoms.calc = calc_old

        return new_atoms

    def minimise(self, atoms, **kwargs):

        new_atoms = self.run(atoms, **kwargs)

        # read lasp.out get opt info
        with open(self._directory_path / "lasp.out", "r") as fopen:
            lines = fopen.readlines()
        opt_indices = []
        for i, line in enumerate(lines):
            if line.strip().startswith("Allopt"):
                opt_indices.append(i)
        final_step_info = lines[opt_indices[-2]+2:opt_indices[-1]-1]
        min_results = "".join(final_step_info)
 
        return new_atoms, min_results


class LaspNN(FileIOCalculator):

    name = "LaspNN"
    implemented_properties = ["energy", "forces"]
    # implemented_propertoes = ["energy", "forces", "stress"]
    command = "lasp"

    default_parameters = {
        # built-in parameters
        "potential": "NN",
        "explore_type": "ssw", # nve nvt npt rigidssw train
        "SSW.ftol": 0.05, # fmax
        "SSW.SSWsteps": 0, # 0 sp 1 opt >1 ssw search
        "SSW.Bfgs_maxstepsize": 0.2,
        "SSW.MaxOptstep": 0, # lasp default 300
        "SSW.output": "T",
        "SSW.printevery": "T",
        # calculator-related
        "constraint": None, # str, lammps-like notatio
    }

    def __init__(self, *args, label="LASP", **kwargs):
        FileIOCalculator.__init__(self, *args, label=label, **kwargs)

        self._directory_path = Path(self.directory)

        return
    
    def calculate(self, *args, **kwargs):
        self._directory_path = Path(self.directory) # NOTE: check if directory changed before calc
        FileIOCalculator.calculate(self, *args, **kwargs)

        return
    
    def write_input(self, atoms, properties=None, system_changes=None):
        # create calc dir
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # structure
        write(
            self._directory_path / "lasp.str", atoms, format="dmol-arc",
            parallel=False
        )

        # check symbols and corresponding potential file
        atomic_types = set(self.atoms.get_chemical_symbols()) # TODO: sort by 

        # input
        content  = "potential {}\n".format(self.parameters["potential"])
        content += "%block netinfo\n"
        for atype in atomic_types:
            # write path
            pot_path = Path(self.parameters["pot"][atype]).resolve()
            content += "  {:<4s} {:s}\n".format(atype, pot_path.name)
            # creat potential link
            pot_link = self._directory_path / pot_path.name
            if not pot_link.is_symlink(): # false if not exists
                pot_link.symlink_to(pot_path)
        content += "%endblock netinfo\n"

        constraint = self.parameters["constraint"]
        if constraint is not None:
            content += "%block fixatom\n"
            cons_block = constraint.strip().split()
            for block in cons_block:
                s, e = block.split(":")
                content += "  {} {} xyz\n".format(s, e)
            content += "%endblock fixatom\n"

        explore_type = self.parameters["explore_type"]
        content += "\nexplore_type {}\n".format(explore_type)
        if explore_type == "ssw":
            for key, value in self.parameters.items():
                if key.startswith("SSW."):
                    content += "{}  {}\n".format(key, value)
        elif explore_type in ["nve", "nvt", "npt"]:
            content += "explore_type nve\n"
            content += "MD.dt           1.0\n"
            content += "MD.ttotal       0\n"
            content += "MD.target_T     300\n"
            content += "MD.realmass     .true.\n"
            content += "MD.print_freq      10\n"
        else:
            # TODO: should check explore_type in init
            pass

        with open(self._directory_path / "lasp.in", "w") as fopen:
            fopen.write(content)

        return
    
    def read_results(self):
        """read LASP results"""
        natoms = len(self.atoms)

        # have to read last structure
        with open(self._directory_path / "allfor.arc", "r") as fopen:
            while True:
                line = fopen.readline()
                if line.startswith(" For"):
                    energy = float(line.split()[3])
                    # stress
                    line = fopen.readline()
                    stress = np.array(line.split()) # TODO: what is the format of stress
                    # forces
                    forces = []
                    for j in range(natoms):
                        line = fopen.readline()
                        forces.append(line.split())
                    forces = np.array(forces, dtype=float)
                    cur_results = {"energy": energy, "forces": forces}
                if line.strip() == "":
                    cur_results = []
                if not line: # if line == "":
                    break

        self.results["energy"] = energy
        self.results["forces"] = forces

        return


if __name__ == "__main__":
    atoms = read("/mnt/scratch2/users/40247882/catsign/lasp-main/xxx.xyz")
    pot_path = "/mnt/scratch2/users/40247882/catsign/lasp-main/ZnCrO.pot"
    pot = dict(
        O  = pot_path,
        Cr = pot_path,
        Zn = pot_path
    )
    calc = LaspNN(
        directory = "./LaspNN-Worker",
        command = "mpirun -n 4 lasp",
        pot=pot
    )

    atoms.calc = calc
    print("initial energy: ", atoms.get_potential_energy())
    # print(atoms.get_forces())

    constraint = "1:24 49:72"

    # use LaspDynamics
    st = time.time()
    worker = LaspDynamics(calc, directory=calc.directory)
    new_atoms, min_results = worker.minimise(atoms, fmax=0.05, steps=10, constraint=constraint)
    et = time.time()
    #print(new_atoms.get_forces())
    print(new_atoms.get_potential_energy())
    print(min_results)
    print("time: ", et - st)

    exit()

    # use ASE internal dynamics
    from GDPy.calculator.ase_interface import AseDynamics
    st = time.time()
    worker = AseDynamics(calc, directory=calc.directory)
    #worker.run(atoms, fmax=0.05, steps=10)
    _ = worker.minimise(atoms, fmax=0.05, steps=10)
    et = time.time()
    print("time: ", et - st)
    print(_)
    print("opt energy: ", atoms.get_potential_energy())