#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import time
import shutil
import warnings
from pathlib import Path

import numpy as np


from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError

from GDPy.calculator.dynamics import AbstractDynamics

from ase.calculators.singlepoint import SinglePointCalculator

"""
output files by lasp NVE-MD
    allfor.arc  allkeys.log  allstr.arc  firststep.restart
    md.arc  md.restart  vel.arc
"""

def read_laspset(train_structures):
    """ read lasp TrainStr.txt and TrainFor.txt
    """
    train_structures = Path(train_structures)
    frames = []

    all_energies, all_forces, all_stresses = [], [], []

    # - TrainStr.txt
    # TODO: use yield
    with open(train_structures, "r") as fopen:
        while True:
            line = fopen.readline()
            if line.strip().startswith("Start one structure"):
                # - energy
                line = fopen.readline()
                energy = float(line.strip().split()[-2])
                all_energies.append(energy)
                # - natoms
                line = fopen.readline()
                natoms = int(line.strip().split()[-1])
                # skip 5 lines, symbol info and training weights
                skipped_lines = [fopen.readline() for i in range(5)]
                # - cell
                cell = np.array([fopen.readline().strip().split()[1:] for i in range(3)], dtype=float)
                # - symbols, positions, and charges
                anumbers, positions, charges = [], [], []
                for i in range(natoms):
                    data = fopen.readline().strip().split()[1:]
                    anumbers.append(int(data[0]))
                    positions.append([float(x) for x in data[1:4]])
                    charges.append(float(data[-1]))
                atoms = Atoms(numbers=anumbers, positions=positions, cell=cell, pbc=True)
                assert fopen.readline().strip().startswith("End one structure")
                frames.append(atoms)
                #break
            if not line:
                break
    
    # - TrainFor.txt
    train_forces = train_structures.parent / "TrainFor.txt"
    with open(train_forces, "r") as fopen:
        while True:
            line = fopen.readline()
            if line.strip().startswith("Start one structure"):
                # - stress, voigt order
                stress = np.array(fopen.readline().strip().split()[1:], dtype=float)
                # - symbols, forces
                anumbers, forces = [], []
                line = fopen.readline()
                while True:
                    if line.strip().startswith("force"):
                        data = line.strip().split()[1:]
                        anumbers.append(int(data[0]))
                        forces.append([float(x) for x in data[1:4]])
                    else:
                        all_forces.append(forces)
                        assert line.strip().startswith("End one structure")
                        break
                    line = fopen.readline()
                #break
            if not line:
                break
    
    for i, atoms in enumerate(frames):
        calc = SinglePointCalculator(
            atoms, energy=all_energies[i], forces=all_forces[i]
        )
        atoms.calc = calc
    write(train_structures.parent / "dataset.xyz", frames)

    return frames


class LaspDynamics(AbstractDynamics):

    """ local optimisation
    """

    saved_cards = ["allstr.arc", "allfor.arc"]

    method = "opt"

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

    def minimise(self, atoms, repeat=1, extra_info=None, **kwargs) -> Atoms:
        """ return a new atoms with singlepoint calc
            input atoms wont be changed
        """
        # TODO: add verbose
        print(f"\nStart minimisation maximum try {repeat} times...")
        for i in range(repeat):
            print("attempt ", i)
            min_atoms = self.run(atoms, **kwargs)
            min_results = self.__read_min_results(self._directory_path / "lasp.out")
            print(min_results)
            # add few information
            if extra_info is not None:
                min_atoms.info.update(extra_info)
            maxforce = np.max(np.fabs(min_atoms.get_forces(apply_constraint=True)))
            if maxforce <= kwargs["fmax"]:
                break
            else:
                atoms = min_atoms
                print("backup old data...")
                for card in self.saved_cards:
                    card_path = self._directory_path / card
                    bak_fmt = ("bak.{:d}."+card)
                    idx = 0
                    while True:
                        bak_card = bak_fmt.format(idx)
                        if not Path(bak_card).exists():
                            saved_card_path = self._directory_path / bak_card
                            shutil.copy(card_path, saved_card_path)
                            break
                        else:
                            idx += 1
        else:
            warnings.warn(f"Not converged after {repeat} minimisations, and save the last atoms...", UserWarning)

        return min_atoms
    
    def __read_min_results(self, fpath):
        # read lasp.out get opt info
        with open(fpath, "r") as fopen:
            lines = fopen.readlines()
        opt_indices = []
        for i, line in enumerate(lines):
            if line.strip().startswith("Allopt"):
                opt_indices.append(i)
        final_step_info = lines[opt_indices[-2]+2:opt_indices[-1]-1]
        min_results = "".join(final_step_info)

        return min_results


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
        "constraint": None, # str, lammps-like notation
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
    # ===== test lasp format =====
    read_laspset("/mnt/scratch2/users/40247882/pbe-oxides/LASPset/TrainStr.txt")
    exit()

    # ===== test lasp calculator =====
    # atoms = read("/mnt/scratch2/users/40247882/catsign/lasp-main/xxx.xyz")
    atoms = read("/mnt/scratch2/users/40247882/catsign/lasp-main/ga-surface/PGM.xyz")

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

    # constraint = "1:24 49:72"
    constraint = "1:12 51:62"

    # use LaspDynamics
    st = time.time()
    worker = LaspDynamics(calc, directory=calc.directory)
    new_atoms, min_results = worker.minimise(atoms, fmax=0.05, steps=100, constraint=constraint)
    et = time.time()
    #print(new_atoms.get_forces())
    print(new_atoms.get_potential_energy())
    print(min_results)
    print("time: ", et - st)

    write("PGM_opt.xyz", new_atoms)

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