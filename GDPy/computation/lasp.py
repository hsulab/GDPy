#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import time
import shutil
import warnings
from pathlib import Path
from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.computation.driver import AbstractDriver
from GDPy.builder.constraints import parse_constraint_info


"""Driver and calculator of LaspNN.

Output files by LASP NVE-MD are 
    allfor.arc  allkeys.log  allstr.arc  firststep.restart 
    md.arc  md.restart  vel.arc

"""

class LaspEnergyError(Exception):
    """Error due to the failure of LASP energy.

    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def read_laspset(train_structures):
    """Read LASP TrainStr.txt and TrainFor.txt files."""
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


class LaspDriver(AbstractDriver):

    """Driver for LASP."""

    name = "lasp"

    # - defaults
    default_task = "min"
    supported_tasks = ["min", "md"]

    default_init_params = {
        "min": {
            "explore_type": "ssw",
            #"SSW.SSWsteps": 1 # min_style
            "min_style": 1, # SSW.SSWsteps
            "dump_period": 1
        },
        "md": {
            "md_style": "nvt",
            "velocity_seed": None,
            "timestep": 1.0, # fs
            "temp": 300, # K
            "Tdamp": 100, # fs
            "pres": 1.0, # atm
            "Pdamp": 100,
            "dump_period": 1
        }
    }

    default_run_params = {
        "min": {
            "SSW.ftol": 0.05,
            "SSW.MaxOptstep": 0
        },
        "md": {
            "SSW.MaxOptstep": 0
        }
    }

    param_mapping = {
        # - min
        "min_style": "SSW.SSWsteps",
        # - md
        "md_style": "explore_type",
        "timestep" : "MD.dt", # fs
        "dump_period": "MD.print_freq", # for MD.print_strfreq as well
        # MD.ttotal uses value equal SSW.MaxOptstep*MD.dt
        "temp": "MD.target_T", # K
        "Tdamp": "nhmass",
        "pres": "MD.target_P",
        "Pdamp": "MD.prmass",
        "fmax": "SSW.ftol",
        "steps": "SSW.MaxOptstep",
        "velocity_seed": "Ranseed"
    }

    #: List of output files would be saved when restart.
    saved_cards = ["allstr.arc", "allfor.arc"]

    def _parse_params(self, params: dict):
        """Set several connected parameters."""
        super()._parse_params(params)

        return 

    def __set_special_params(self, params: dict) -> dict:
        """Set several connected parameters."""
        if self.task == "md":
            total_time = params.get("MD.ttotal", None)
            if total_time is None:
                params["MD.ttotal"] = params["MD.dt"]*params["SSW.MaxOptstep"]
        
            init_temp = params.get("MD.initial_T", None)
            if init_temp is None:
                params["MD.initial_T"] = params["MD.target_T"]

            params["MD.print_freq"] *= params["MD.dt"] # freq has unit fs
            params["MD.print_strfreq"] = params["MD.print_freq"]

        return params

    def run(self, atoms_, read_exists: bool=True, extra_info: dict=None, *args, **kwargs) -> Atoms:
        """Run the driver."""
        atoms = atoms_.copy()

        # - backup calc params
        calc_old = atoms.calc
        params_old = copy.deepcopy(self.calc.parameters)

        # set special keywords
        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)

        # - run params
        kwargs = self._map_params(kwargs)

        run_params = self.run_params.copy()
        run_params.update(kwargs)

        # - init params
        run_params.update(**self.init_params)

        run_params = self.__set_special_params(run_params)

        self.calc.set(**run_params)
        atoms.calc = self.calc

        # - run dynamics
        try:
            if read_exists:
                converged = atoms.calc._is_converged()
                if converged:
                    # atoms.calc.read_results()
                    pass
                else:
                    # NOTE: restart calculation!!!
                    _  = atoms.get_forces()
                    converged = atoms.calc._is_converged()
            else:
                _  = atoms.get_forces()
                converged = atoms.calc._is_converged()
        except OSError:
            converged = False
        #else:
        #    converged = True

        assert converged, "LaspDriver is not converged..."
        
        # read new atoms positions
        traj_frames = self.calc._read_trajectory()
        new_atoms = traj_frames[-1]

        # - restore to old calculator
        self.calc.reset() 
        self.calc.parameters = params_old
        if calc_old is not None:
            atoms.calc = calc_old

        return new_atoms
    
    def read_trajectory(self, add_step_info=True, *args, **kwargs) -> List[Atoms]:
        """Read trajectory in the current working directory."""
        return self.calc._read_trajectory(add_step_info)


class LaspNN(FileIOCalculator):

    #: Calculator name.
    name: str = "LaspNN"

    #: Implemented properties.
    implemented_properties: List[str] = ["energy", "forces"]
    # implemented_propertoes = ["energy", "forces", "stress"]

    #: LASP command.
    command = "lasp"

    #: Default calculator parameters, NOTE which have ase units.
    default_parameters = {
        # built-in parameters
        "potential": "NN",
        # - general settings
        "explore_type": "ssw", # ssw, nve nvt npt rigidssw train
        "Ranseed": None,
        # - ssw
        "SSW.internal_LJ": True,
        "SSW.ftol": 0.05, # fmax
        "SSW.SSWsteps": 0, # 0 sp 1 opt >1 ssw search
        "SSW.Bfgs_maxstepsize": 0.2,
        "SSW.MaxOptstep": 0, # lasp default 300
        "SSW.output": "T",
        "SSW.printevery": "T",
        # md
        "MD.dt": 1.0, # fs
        "MD.ttotal": 0, 
        "MD.initial_T": None,
        "MD.equit": 0,
        "MD.target_T": 300, # K
        "MD.nhmass": 1000, # eV*fs**2
        "MD.target_P": 300, # 1 bar = 1e-4 GPa
        "MD.prmass": 1000, # eV*fs**2
        "MD.realmass": ".true.",
        "MD.print_freq": 10,
        "MD.print_strfreq": 10,
        # calculator-related
        "constraint": None, # str, lammps-like notation
    }

    def __init__(self, *args, label="LASP", **kwargs):
        """Init calculator.

        The potential path would be resolved.

        """
        FileIOCalculator.__init__(self, *args, label=label, **kwargs)

        # NOTE: need resolved pot path
        pot_ = {}
        pot = self.parameters.get("pot", None)
        for k, v in pot.items():
            pot_[k] = Path(v).resolve()
        self.set(pot=pot_)

        return
    
    def calculate(self, *args, **kwargs):
        """Perform the calculation."""
        FileIOCalculator.calculate(self, *args, **kwargs)

        return
    
    def write_input(self, atoms, properties=None, system_changes=None):
        """Write LASP inputs."""
        # create calc dir
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # structure
        write(
            os.path.join(self.directory, "lasp.str"), atoms, format="dmol-arc",
            parallel=False
        )

        # check symbols and corresponding potential file
        atomic_types = set(self.atoms.get_chemical_symbols()) # TODO: sort by 

        # - potential choice
        # NOTE: only for LaspNN now
        content  = "potential {}\n".format(self.parameters["potential"])
        assert self.parameters["potential"] == "NN", "Lasp calculator only support NN now."

        content += "%block netinfo\n"
        for atype in atomic_types:
            # write path
            pot_path = Path(self.parameters["pot"][atype]).resolve()
            content += "  {:<4s} {:s}\n".format(atype, pot_path.name)
            # creat potential link
            pot_link = Path(os.path.join(self.directory,pot_path.name))
            if not pot_link.is_symlink(): # false if not exists
                pot_link.symlink_to(pot_path)
        content += "%endblock netinfo\n"

        # - atom constraint
        constraint = self.parameters["constraint"]
        mobile_text, frozen_text = parse_constraint_info(atoms, constraint)

        if frozen_text is not None:
            content += "%block fixatom\n"
            frozen_block = frozen_text.strip().split()
            for block in frozen_block:
                info = block.split(":")
                if len(info) == 2:
                    s, e = info
                else:
                    s, e = info[0], info[0]
                content += "  {} {} xyz\n".format(s, e)
            content += "%endblock fixatom\n"
        
        # - general settings
        seed = self.parameters["Ranseed"]
        if seed is None:
            seed = np.random.randint(1,10000)
        content += "{}  {}".format("Ranseed", seed)

        # - simulation task
        explore_type = self.parameters["explore_type"]
        content += "\nexplore_type {}\n".format(explore_type)
        if explore_type == "ssw":
            for key, value in self.parameters.items():
                if key.startswith("SSW."):
                    content += "{}  {}\n".format(key, value)
        elif explore_type in ["nve", "nvt", "npt"]:
            required_keys = [
                "MD.dt", "MD.ttotal", "MD.realmass", 
                "MD.print_freq", "MD.print_strfreq"
            ]
            if explore_type == "nvt":
                required_keys.extend(["MD.initial_T", "MD.target_T", "MD.equit"])
            if explore_type == "npt":
                required_keys.extend(["MD.target_P"])
            
            self.parameters["MD.ttotal"] = (
                self.parameters["MD.dt"] * self.parameters["SSW.MaxOptstep"]
            )

            for k, v in self.parameters.items():
                if k == "MD.target_P":
                    v *= 1e4 # from bar to GPa
                if k in required_keys:
                    content += "{}  {}\n".format(k, v)
        else:
            # TODO: should check explore_type in init
            pass

        with open(os.path.join(self.directory, "lasp.in"), "w") as fopen:
            fopen.write(content)

        return
    
    def read_results(self):
        """Read LASP results."""
        # have to read last structure
        traj_frames = self._read_trajectory()

        energy = traj_frames[-1].get_potential_energy()
        forces = traj_frames[-1].get_forces().copy()

        self.results["energy"] = energy
        self.results["forces"] = forces

        return
    
    def _is_converged(self) -> bool:
        """Check whether LASP simulation is converged."""
        converged = False
        lasp_out = Path(os.path.join(self.directory, "lasp.out" ))
        if lasp_out.exists():
            with open(lasp_out, "r") as fopen:
                lines = fopen.readlines()
            if (# NOTE: its a typo in LASP!!
                lines[-1].strip().startswith("elapse_time") or # v3.3.4
                lines[-1].strip().startswith("Elapse_time")    # v3.4.5
            ):
                converged = True

        return converged
    
    def _read_trajectory(self, add_step_info: bool=True) -> List[Atoms]:
        """Read simulation trajectory."""
        traj_frames = read(os.path.join(self.directory, "allstr.arc"), ":", format="dmol-arc")
        natoms = len(traj_frames[-1])

        traj_steps = []
        traj_energies = []
        traj_forces = []

        # NOTE: for lbfgs opt, steps+2 frames would be output?

        # have to read last structure
        with open(os.path.join(self.directory, "allfor.arc"), "r") as fopen:
            while True:
                line = fopen.readline()
                if line.strip().startswith("For"):
                    step = int(line.split()[1])
                    traj_steps.append(step)
                    # NOTE: need check if energy is a number
                    #       some ill structure may result in large energy of ******
                    energy_data = line.split()[3]
                    try:
                        energy = float(energy_data)
                    except ValueError:
                        energy = np.inf
                        msg = "Energy is too large at {}. The structure maybe ill-constructed.".format(self.directory)
                        warnings.warn(msg, UserWarning)
                    traj_energies.append(energy)
                    # stress
                    line = fopen.readline()
                    stress = np.array(line.split()) # TODO: what is the format of stress
                    # forces
                    forces = []
                    for j in range(natoms):
                        line = fopen.readline()
                        force_data = line.strip().split()
                        if len(force_data) == 3: # expect three numbers
                            pass
                        else: # too large forces make out become ******
                            force_data = [np.inf]*3
                        forces.append(force_data)
                    forces = np.array(forces, dtype=float)
                    traj_forces.append(forces)
                if line.strip() == "":
                    pass
                if not line: # if line == "":
                    break
        assert len(traj_frames) == len(traj_steps), "Output number is inconsistent."
        
        # - create traj
        for i, atoms in enumerate(traj_frames):
            calc = SinglePointCalculator(
                atoms, energy=traj_energies[i], forces=traj_forces[i]
            )
            atoms.calc = calc
        
        if add_step_info:
            for step, atoms in zip(traj_steps, traj_frames):
                atoms.info["step"] = step

        return traj_frames


if __name__ == "__main__":
    pass
