"""An ASE calculator interface.

"""

import os
import copy
import shutil
import warnings
import subprocess
import pathlib
from pathlib import Path
import dataclasses

from collections.abc import Iterable
from typing import List, Mapping, Dict, Optional, NoReturn

import numpy as np

from ase import Atoms
from ase import units
from ase.data import atomic_numbers, atomic_masses
from ase.io import read, write
from ase.io.lammpsrun import read_lammps_dump_text
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.calculators.calculator import (
    CalculationFailed,
    Calculator, all_changes, PropertyNotImplementedError, FileIOCalculator
)
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lammps import unitconvert

from GDPy.calculator.dynamics import AbstractDynamics
from GDPy.utils.command import find_backups

from GDPy.builder.constraints import parse_constraint_info

dataclasses.dataclass(frozen=True)
class AseLammpsSettings:

    inputstructure_filename = "stru.data"
    trajectory_filename = "surface.dump"
    log_filename = "log.lammps"
    deviation_filename = "model_devi.out"


ASELMPCONFIG = AseLammpsSettings()

def parse_thermo_data(logfile_path) -> dict:
    """"""
    # - read thermo data
    # better move to calculator class
    with open(logfile_path, "r") as fopen:
        lines = fopen.readlines()
    start_idx, end_idx = None, None
    for idx, line in enumerate(lines):
        if line.startswith("Step"):
            start_idx = idx
        # TODO: if not finish?
        if line.startswith("Loop time"):
            end_idx = idx
        if start_idx is not None and end_idx is not None:
            break
    else:
        raise ValueError("error in lammps output.")
    #loop_time = lines[end_idx].strip().split()[3]
    loop_time = lines[end_idx]
    # -- parse index of PotEng
    # TODO: save timestep info?
    thermo_keywords = lines[start_idx].strip().split()
    if "PotEng" not in thermo_keywords:
        raise ValueError("cant find PotEng in lammps output.")
    thermo_data = lines[start_idx+1:end_idx]
    thermo_data = np.array([line.strip().split() for line in thermo_data], dtype=float).transpose()
    #print(thermo_data)
    thermo_dict = {}
    for i, k in enumerate(thermo_keywords):
        thermo_dict[k] = thermo_data[i]

    return thermo_dict, loop_time

class LmpDynamics(AbstractDynamics):

    """ use lammps to perform dynamics
        minimisation and/or molecular dynamics
    """

    delete = []
    keyword: Optional[str] = None
    # special_keywords: Dict[str, str] = dict()
    special_keywords = {
        'fmax': '{}',
        'steps': 'maxcycle={}',
    }

    saved_cards = ["surface.dump"]

    def __init__(self, calc=None, dynrun_params={}, directory="./"):
        """"""
        self.calc = calc
        self.calc.reset()

        self.set_output_path(directory)

        # - parse method
        #self.method = dynrun_params.get("method", "min")
        self.dynrun_params = dynrun_params

        return
    
    def reset(self):
        """ remove calculated quantities
        """
        self.calc.reset()

        return
    
    def set_output_path(self, directory):
        """"""
        self._directory_path = pathlib.Path(directory)
        self.calc.directory = pathlib.Path(directory)

        return
    
    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)
        
        return

    def set_keywords(self, kwargs):
        """"""
        args = kwargs.pop(self.keyword, [])
        if isinstance(args, str):
            args = [args]
        elif isinstance(args, Iterable):
            args = list(args)

        for key, template in self.special_keywords.items():
            if key in kwargs:
                val = kwargs.pop(key)
                args.append(template.format(val))

        kwargs[self.keyword] = args

        return
    
    def run(self, atoms, read_exists: bool=True, extra_info: dict=None, **kwargs):
        """"""
        # - backup old params
        # TODO: change to context message?
        calc_old = atoms.calc 
        params_old = copy.deepcopy(self.calc.parameters)

        # - set special keywords
        # self.delete_keywords(kwargs)
        # self.delete_keywords(self.calc.parameters)
        # self.set_keywords(kwargs)

        # - convert units if necessary
        #print(self.dynrun_params)
        new_params = self.dynrun_params.copy()
        #print("new_params: ", new_params)
        new_params.update(**kwargs)
        #print("new_params: ", new_params)

        self.calc.set(**new_params)
        atoms.calc = self.calc

        # - run dynamics
        try:
            # NOTE: some calculation can overwrite existed data
            if read_exists:
                is_finished, wall_time = atoms.calc._is_finished()
                if is_finished:
                    print(f"found finished {self._directory_path.name}.")
                else:
                    # TODO: restart calculation!!!
                    _  = atoms.get_forces()
            else:
                _  = atoms.get_forces()
        except OSError:
            converged = False
        else:
            converged = True

        # NOTE: always use dynamics calc
        # TODO: should change positions and other properties for input atoms?
        assert atoms.calc.cached_traj_frames is not None, "failed to read results in lammps"
        new_atoms = atoms.calc.cached_traj_frames[-1]
        self.cached_loop_time = atoms.calc.cached_loop_time

        if extra_info is not None:
            new_atoms.info.update(**extra_info)

        # - reset params
        self.calc.parameters = params_old
        self.calc.reset()
        if calc_old is not None:
            atoms.calc = calc_old

        return new_atoms

    def minimise(self, atoms, read_exists: bool = True, read_backup: bool = False, repeat=1, extra_info=None, **kwargs) -> Atoms:
        """ return a new atoms with singlepoint calc
            input atoms wont be changed
        """
        # TODO: add verbose
        print(f"\nStart minimisation maximum try {repeat} times...")
        for i in range(repeat):
            print("attempt ", i)
            min_atoms = self.run(atoms, read_exists=read_exists, **kwargs)
            min_results = self.__read_min_results(self._directory_path / "log.lammps")
            min_results += self.cached_loop_time
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
        
        # gather trajectories
        if read_backup:
            backups = find_backups(self._directory_path, self.saved_cards[0])
            frames = read(
                backups[0], ":", "lammps-dump-text", 
                specorder=self.calc.specorder, units=self.calc.units
            )
            for bak in backups[1:]:
                frames.extend(
                    read(
                        bak, ":", "lammps-dump-text", 
                        specorder=self.calc.specorder, units=self.calc.units
                    )[1:]
                )
        
            write(self._directory_path/"merged_traj.xyz", frames)
        
        return min_atoms

    def __read_min_results(self, fpath):
        # read energy
        with open(fpath, "r") as fopen:
            lines = fopen.readlines()
        for idx, line in enumerate(lines):
            if line.startswith("Minimization stats:"):
                stat_idx = idx
                break
        else:
            raise ValueError('error in lammps minimization.')
        stat_content = "".join(lines[stat_idx:stat_idx+9])

        return stat_content
    



class Lammps(FileIOCalculator):

    name = "Lammps"
    supported_pairstyles = ["deepmd", "eann", "reax/c"]
    implemented_properties = ["energy", "forces", "stress"]

    # only for energy and forces, eV and eV/AA
    CONVERTOR = {
        "metal": 1.0,
        "real": units.kcal / units.mol
    }

    command = "lmp 2>&1 > lmp.out"

    # NOTE: here all params are in ase unit
    default_parameters = dict(
        # ase params
        steps = 0,
        fmax = 0.05, # eV, for min
        constraint = None, # index of atoms, start from 0
        method = "min",
        # --- lmp params ---
        units = "metal",
        atom_style = "atomic",
        processors = "* * 1",
        boundary = "p p p",
        pair_style = None,
        pair_coeff = None,
        mass = "* 1.0",
        dump_period = 1,
        # - md
        md_style = "nvt",
        timestep = 1.0, # fs
        temp = 300,
        pres = 1.0,
        Tdamp = 100, # fs
        Pdamp = 100,
        # - minimisation
        min_style = "fire",
        min_modify = None
    )

    specorder = None
    cached_traj_frames = None
    cached_loop_time = None

    def __init__(
        self, 
        command = None, 
        label = name, 
        #pair_style: Mapping = None, # pair_style specific parameters
        **kwargs
    ):
        """"""
        FileIOCalculator.__init__(self, command=command, label=label, **kwargs)

        self.command = command
        
        # TODO: this should be shortcuts for built-in potentials
        #self.pair_style = pair_style
        #style = pair_style["model"]
        #if style == "reax/c":
        #    self.units = "real"
        #    self.atom_style = "charge"
        #elif style == "deepmd":
        #    self.units = "metal"
        #    self.atom_style = "atomic"
        #elif style == "eann":
        #    self.units = "metal"
        #    self.atom_style = "atomic"

        # - check potential
        assert self.pair_style is not None, "pair_style is not set."

        return
    
    def __getattr__(self, key):
        """ Corresponding getattribute-function 
        """
        if key != "parameters" and key in self.parameters:
            return self.parameters[key]
        return object.__getattribute__(self, key)
    
    def calculate(self, atoms=None, properties=['energy'],
            system_changes=all_changes):
        # check specorder
        self.check_specorder(atoms)

        # init for creating the directory
        FileIOCalculator.calculate(self, atoms, properties, system_changes)

        return
    
    def check_specorder(self, atoms):
        """check specorder for read and write structure of lammps"""
        # elements
        specorder = list(set(atoms.get_chemical_symbols()))
        specorder.sort() # by alphabet

        self.specorder = specorder

        return
    
    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # write structure
        stru_data = os.path.join(self.directory, ASELMPCONFIG.inputstructure_filename)
        write_lammps_data(
            stru_data, atoms, specorder=self.specorder, force_skew=True, 
            units=self.units, atom_style=self.atom_style
        )

        # write input
        self._write_input(atoms)

        return
    
    def _is_finished(self):
        """ check whether the simulation is finished
        """
        is_finished, wall_time = False, "not finished"
        log_filepath = Path(os.path.join(self.directory, ASELMPCONFIG.log_filename))

        if log_filepath.exists():
            END_FLAG = "Total wall time:"
            with open(log_filepath, "r") as fopen:
                lines = fopen.readlines()
        
            for line in lines:
                if line.strip().startswith(END_FLAG):
                    is_finished = True
                    wall_time = line.strip().split()[-1]
                    break
            else:
                is_finished = False
        else:
            is_finished = False

        return is_finished, wall_time
    
    def read_results(self):
        """"""
        # obtain results
        self.results = {}

        # - Be careful with UNITS
        # read forces from dump file
        self.cached_traj_frames = self._read_trajectory()
        #traj_frames = read(
        #    os.path.join(self.directory, ASELMPCONFIG.trajectory_filename), ":", "lammps-dump-text", 
        #    specorder=self.specorder, units=self.units
        #)
        #nframes = len(traj_frames)

        ## - only create last frame to atoms
        #dump_atoms = traj_frames[-1]
        #self.results["forces"] = unitconvert.convert(dump_atoms.get_forces(), "force", self.units, "ASE")

        ## read energy
        #thermo_dict = parse_thermo_data(os.path.join(self.directory, ASELMPCONFIG.log_filename))
        #energy = thermo_dict["PotEng"][nframes-1]

        #self.results["energy"] = unitconvert.convert(energy, "energy", self.units, "ASE")

        self.results["forces"] = self.cached_traj_frames[-1].get_forces().copy()
        self.results["energy"] = self.cached_traj_frames[-1].get_potential_energy()

        return

    def _read_trajectory(self, label_steps: bool=False) -> List[Atoms]:
        """"""
        # NOTE: always use dynamics calc
        # - parse spec order
        #self.check_specorder(atoms)
        
        # - read trajectory that contains positions and forces
        _directory_path = Path(self.directory)
        # NOTE: forces would be zero if setforce 0 is set
        traj_frames = read(
            _directory_path / ASELMPCONFIG.trajectory_filename, ":", "lammps-dump-text", 
            specorder=self.specorder, units=self.units
        )
        # - read thermo data
        thermo_dict, loop_time = parse_thermo_data(_directory_path / ASELMPCONFIG.log_filename)
        self.cached_loop_time = loop_time

        # NOTE: last frame would not be dumpped if timestep not equals multiple*dump_period
        pot_energies = [unitconvert.convert(p, "energy", self.units, "ASE") for p in thermo_dict["PotEng"][:len(traj_frames)]]
        assert len(pot_energies) == len(traj_frames), "number of pot energies and frames is inconsistent."

        for pot_eng, atoms in zip(pot_energies, traj_frames):
            forces = atoms.get_forces()
            forces = unitconvert.convert(forces, "force", self.units, "ASE")
            sp_calc = SinglePointCalculator(atoms, energy=pot_eng, forces=forces)
            atoms.calc = sp_calc
        
        # - check model_devi.out
        # TODO: convert units?
        devi_path = _directory_path / ASELMPCONFIG.deviation_filename
        if devi_path.exists():
            with open(devi_path, "r") as fopen:
                lines = fopen.readlines()
            dkeys = ("".join([x for x in lines[0] if x != "#"])).strip().split()
            dkeys = [x.strip() for x in dkeys][1:]
            data = np.loadtxt(devi_path, dtype=float).transpose()[1:,:len(traj_frames)]

            for i, atoms in enumerate(traj_frames):
                for j, k in enumerate(dkeys):
                    atoms.info[k] = data[j,i]
        
        # - label steps
        if label_steps:
            for i, atoms in enumerate(traj_frames):
                atoms.info["source"] = _directory_path.name
                atoms.info["step"] = int(thermo_dict["Step"][i])

        return traj_frames
    
    def _write_input(self, atoms) -> NoReturn:
        """"""
        mass_line = ''.join(
            'mass %d %f\n' %(idx+1,atomic_masses[atomic_numbers[elem]]) for idx, elem in enumerate(self.specorder)
        )
        # write in.lammps
        content = ""
        content += "units           %s\n" %self.units
        content += "atom_style      %s\n" %self.atom_style

        # mpi settings
        if self.processors is not None:
            content += "processors {}\n".format(self.processors) # if 2D simulation
        
        # simulation box
        content += "boundary        p p p\n"
        content += "\n"
        content += "box             tilt large\n"
        content += "read_data	    %s\n" %ASELMPCONFIG.inputstructure_filename
        content += "change_box      all triclinic\n"

        # particle masses
        content += mass_line
        content += "\n"

        # pair
        #if pair_style["model"] == "reax/c":
        #    # reaxff uses real unit, force kcal/mol/A
        #    content += "pair_style	reax/c NULL\n"
        #    content += "pair_coeff	* * /users/40247882/projects/oxides/gdp-main/reaxff/ffield.reax.PtO %s\n" %(' '.join(self.specorder))
        #    content += "neighbor        0.0 bin\n"
        #    content += "fix             2 all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
        #    content += "\n"
        #elif pair_style["model"] == "eann":
        #    out_freq = pair_style.get("out_freq", 10)
        #    if out_freq == 10:
        #        style_args = "{}".format(pair_style["file"])
        #    else:
        #        style_args = "{} out_freq {}".format(pair_style["file"], out_freq)
        #    content += "pair_style	eann %s\n" %style_args
        #    content += "pair_coeff	* * double %s\n" %(" ".join(self.specorder))
        #    content += "neighbor        0.0 bin\n"
        #    content += "\n"
        #elif pair_style["model"] == "deepmd":
        #    content += "pair_style	deepmd %s\n" %pair_style["file"]
        #    content += "pair_coeff	\n" 
        #    content += "neighbor        0.0 bin\n"
        #    content += "\n"
        #if self.pair_coeff is not None:
        #    content += "pair_coeff  {}\n".format(self.pair_coeff)
        
        # MLIP specific settings
        # TODO: neigh settings?
        potential = self.pair_style.strip().split()[0]
        if potential == "reax/c":
            assert self.atom_style == "charge", "reax/c should have charge atom_style"
            content += "pair_style  {}\n".format(self.pair_style)
            content += "neighbor        0.0 bin\n"
            content += "fix             reaxqeq all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
        elif potential == "eann":
            pot_data = self.pair_style.strip().split()[1:]
            endp = len(pot_data)
            for ip, p in enumerate(pot_data):
                if p == "out_freq":
                    endp = ip
                    break
            pot_data = pot_data[:endp]
            if len(pot_data) > 1:
                pair_style = "eann {} out_freq {}".format(" ".join(pot_data), self.dump_period)
            else:
                pair_style = "eann {}".format(" ".join(pot_data))
            content += "pair_style  {}\n".format(pair_style)
            # NOTE: make out_freq consistent with dump_period
            if self.pair_coeff is None:
                pair_coeff = "double * *"
            else:
                pair_coeff = self.pair_coeff
            content += "pair_coeff	{} {}\n".format(pair_coeff, " ".join(self.specorder))
            content += "neighbor        0.0 bin\n"
        elif potential == "deepmd":
            content += "pair_style  {}\n".format(self.pair_style)
            content += "neighbor        0.0 bin\n"
        content += "\n"

        # constraint
        mobile_text, frozen_text = parse_constraint_info(atoms, self.constraint)
        content += "group mobile id %s\n" %mobile_text
        content += "\n"
        if frozen_text is not None:
            # content += "region bottom block INF INF INF INF 0.0 %f\n" %zmin # unit A
            content += "group frozen id %s\n" %frozen_text
            content += "fix cons frozen setforce 0.0 0.0 0.0\n"

        # outputs
        # TODO: use more flexible notations
        if self.method == "min":
            content += "thermo_style    custom step pe ke etotal temp press vol fmax fnorm\n"
        elif self.method == "md":
            content += "compute mobileTemp mobile temp\n"
            content += "thermo_style    custom step c_mobileTemp pe ke etotal press vol lx ly lz xy xz yz\n"
        else:
            pass
        content += "thermo          {}\n".format(self.dump_period) 

        # TODO: How to dump total energy?
        content += "dump		1 all custom {} surface.dump id type x y z fx fy fz\n".format(self.dump_period)
        #content += "dump_modify 1 first yes\n"
        content += "\n"
        
        # --- run type
        if self.method == "min":
            # - minimisation
            content += "min_style       {}\n".format(self.min_style)
            content += "min_modify      integrator verlet tmax 4 # see more on lammps doc about min_modify\n"
            content += "minimize        0.0 %f %d %d # energy tol, force tol, step, force step\n" %(
                unitconvert.convert(self.fmax, "force", "ASE", self.units),
                self.steps, 2.0*self.steps
            )
        elif self.method == "md":
            content += "velocity        mobile create {} {}\n".format(self.temp, np.random.randint(0,10000))
        
            if self.md_style == "nvt":
                Tdamp_ = unitconvert.convert(self.Tdamp, "time", "real", self.units)
                content += "fix             thermostat all nvt temp {} {} {}\n".format(
                    self.temp, self.temp, Tdamp_
                )
            elif self.md_style == "npt":
                pres_ = unitconvert.convert(self.pres, "pressure", "ASE", self.units)
                Tdamp_ = unitconvert.convert(self.Tdamp, "time", "real", self.units)
                Pdamp_ = unitconvert.convert(self.Pdamp, "time", "real", self.units)
                content += "fix             thermostat all npt temp {} {} {} aniso {} {} {}\n".format(
                    self.temp, self.temp, Tdamp_, pres_, pres_, Pdamp_
                )
            elif self.md_style == "nve":
                content += "fix             thermostat all nve \n"

            timestep_ = unitconvert.convert(self.timestep, "time", "real", self.units)
            content += "\n"
            content += f"timestep        {timestep_}\n"
            content += f"run             {self.steps}\n"

        in_file = os.path.join(self.directory, "in.lammps")
        with open(in_file, "w") as fopen:
            fopen.write(content)

        return
 

if __name__ == "__main__":
    # test new lammps

    # test
    calc = Lammps(
        command = "lmp_cat -in ./in.lammps 2>&1 > lmp.out",
        directory =  "./LmpMin-worker",
        pair_style = "eann /mnt/scratch2/users/40247882/pbe-oxides/eann-main/m09/ensemble/model-2/eann_latest_lmp_DOUBLE.pt"
    )

    atoms = read("/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m09/ga/rs/uged-calc_candidates.xyz", "0")

    # test dataclass
    from GDPy.expedition.sample_main import MDParams
    dynrun_params = dataclasses.asdict(MDParams())
    worker = LmpDynamics(calc, dynrun_params=dynrun_params, directory="./LmpWorker")
    worker.run(atoms, steps=1, constraint="1:16")

    exit()

    atoms = read("/mnt/scratch2/users/40247882/catsign/eann-main/m01r/ga-surface/cand2.xyz")
    atoms.calc = calc
    print(atoms.get_potential_energy())

    worker = LmpDynamics(calc, directory=calc.directory)
    min_atoms, min_results = worker.minimise(atoms, fmax=0.2, steps=100, constraint="0:12 24:36")
    print(min_atoms)
    print(min_results)
