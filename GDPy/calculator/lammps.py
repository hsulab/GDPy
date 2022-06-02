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
from typing import List, Mapping, Dict, Optional

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
from GDPy.utils.command import find_backups, convert_indices

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
    
    def run(self, atoms, read_exists=False, **kwargs):
        """"""
        # - backup old params
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

        #print(self.calc.parameters)

        # if not self._directory_path.exists():
        #     self._directory_path.mkdir(parents=True)

        # run dynamics
        try:
            if not read_exists:
                _  = atoms.get_forces()
            else:
                self.calc.check_specorder(atoms)
                self.calc.read_results()
        except OSError:
            converged = False
        else:
            converged = True

        # NOTE: always use dynamics calc
        # read optimised atoms
        new_atoms = read(
            self._directory_path / "surface.dump", ':', "lammps-dump-text", 
            specorder=self.calc.specorder, units=self.calc.units
        )[-1]
        sp_calc = SinglePointCalculator(new_atoms, **copy.deepcopy(self.calc.results))
        new_atoms.calc = sp_calc

        # - reset params
        self.calc.parameters = params_old
        self.calc.reset()
        if calc_old is not None:
            atoms.calc = calc_old

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
            min_results = self.__read_min_results(self._directory_path / "log.lammps")
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

    STRUCTURE_FILE = "stru.data"

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
    )

    specorder = None

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
        stru_data = os.path.join(self.directory, self.STRUCTURE_FILE)
        write_lammps_data(
            stru_data, atoms, specorder=self.specorder, force_skew=True, 
            units=self.units, atom_style=self.atom_style
        )

        # write input
        self.__write_lmp_input()

        return
    
    def read_results(self):
        """"""
        # obtain results
        self.results = {}

        # Be careful with UNITS
        # read energy
        with open(os.path.join(self.directory, 'log.lammps'), 'r') as fopen:
            lines = fopen.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('Minimization stats:'):
                stat_idx = idx
                break
        else:
            raise ValueError('error in lammps minimization.')
        self.results["energy"] = float(lines[stat_idx+3].split()[-1]) * self.CONVERTOR[self.units]

        # read forces from dump file
        dump_atoms = read(
            os.path.join(self.directory, 'surface.dump'), ':', 'lammps-dump-text', 
            specorder=self.specorder, units=self.units
        )[-1]
        self.results['forces'] = dump_atoms.get_forces() * self.CONVERTOR[self.units]

        return
    
    def __write_lmp_input(self):
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
        content += "read_data	    %s\n" %self.STRUCTURE_FILE
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
        content += "pair_style  {}\n".format(self.pair_style)
        #if self.pair_coeff is not None:
        #    content += "pair_coeff  {}\n".format(self.pair_coeff)
        
        # MLIP specific settings
        # TODO: neigh settings?
        potential = self.pair_style.strip().split()[0]
        if potential == "reax/c":
            assert self.atom_style == "charge", "reax/c should have charge atom_style"
            content += "neighbor        0.0 bin\n"
            content += "fix             reaxqeq all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
        elif potential == "eann":
            if self.pair_coeff is None:
                pair_coeff = "double * *"
            else:
                pair_coeff = self.pair_coeff
            content += "pair_coeff	{} {}\n".format(pair_coeff, " ".join(self.specorder))
            content += "neighbor        0.0 bin\n"
        elif potential == "deepmd":
            content += "neighbor        0.0 bin\n"
        content += "\n"

        # constraint
        constraint = self.parameters["constraint"] # lammps convention
        if constraint is not None:
            # content += "region bottom block INF INF INF INF 0.0 %f\n" %zmin # unit A
            content += "group frozen id %s\n" %constraint
            content += "fix cons frozen setforce 0.0 0.0 0.0\n"
            frozen_indices = convert_indices(constraint)
            mobile_indices = [x for x in range(1,len(self.atoms)+1) if x not in frozen_indices]
            mobile_text = convert_indices(mobile_indices)
            content += "group mobile id %s\n" %mobile_text
            content += "\n"
        else:
            mobile_indices = [x+1 for x in range(len(self.atoms))]
            mobile_text = convert_indices(mobile_indices)
            content += "group mobile id %s\n" %mobile_text
            content += "\n"

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
        content += "\n"
        
        # --- run type
        if self.method == "min":
            # - minimisation
            content += "min_style       fire\n"
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
        
        exit()

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
