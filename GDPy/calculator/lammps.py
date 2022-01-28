"""An ASE calculator interface.

"""

import os
import copy
import subprocess
import pathlib

from collections.abc import Iterable
from typing import List, Mapping, Dict, Optional

import numpy as np

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

class LmpDynamics():

    delete = []
    keyword: Optional[str] = None
    # special_keywords: Dict[str, str] = dict()
    special_keywords = {
        'fmax': '{}',
        'steps': 'maxcycle={}',
    }


    def __init__(self, calc=None, directory="./"):

        self.calc = calc
        self.calc.reset()

        self.set_output_path(directory)

        return
    
    def reset(self):
        """ remove calculated quantities
        """
        self.calc.reset()

        return
    
    def set_output_path(self, directory):
        """"""
        self._directory_path = pathlib.Path(directory)

        return
    
    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)

    def set_keywords(self, kwargs):
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
    
    def run(self, atoms, **kwargs):
        """"""
        calc_old = atoms.calc
        params_old = copy.deepcopy(self.calc.parameters)

        # set special keywords
        # self.delete_keywords(kwargs)
        # self.delete_keywords(self.calc.parameters)
        # self.set_keywords(kwargs)

        self.calc.set(**kwargs)
        atoms.calc = self.calc

        # if not self._directory_path.exists():
        #     self._directory_path.mkdir(parents=True)

        # run dynamics
        try:
            _  = atoms.get_forces()
        except OSError:
            converged = False
        else:
            converged = True

        # NOTE: always use dynamics calc
        # back up atoms
        # self.calc.parameters = params_old
        # self.calc.reset()
        # if calc_old is not None:
        #     atoms.calc = calc_old

        return atoms
    
    def minimise(self, atoms, **kwargs):
        """ compatibilty to lammps
        """
        min_atoms = self.run(atoms, **kwargs)

        # read energy
        with open(self._directory_path / "log.lammps", "r") as fopen:
            lines = fopen.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('Minimization stats:'):
                stat_idx = idx
                break
        else:
            raise ValueError('error in lammps minimization.')
        stat_content = ''.join(lines[stat_idx:stat_idx+9])

        return min_atoms, stat_content


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

    default_parameters = {
        "steps": 0,
        "fmax": 0.05, # eV, for min
        "constraint": None # index of atoms, start from 0
    }

    specorder = None

    def __init__(
        self, 
        command = None, 
        label = name, 
        pair_style: Mapping = None, # pair_style specific parameters
        **kwargs
    ):
        """"""
        FileIOCalculator.__init__(self, command=command, label=label, **kwargs)

        self.command = command
        
        self.pair_style = pair_style
        style = pair_style["model"]
        if style == "reax/c":
            self.units = "real"
            self.atom_style = "charge"
        elif style == "deepmd":
            self.units = "metal"
            self.atom_style = "atomic"
        elif style == "eann":
            self.units = "metal"
            self.atom_style = "atomic"

        return
    
    def calculate(self, atoms=None, properties=['energy'],
            system_changes=all_changes):
        # check specorder
        self.__check_specorder(atoms)

        # init for creating the directory
        FileIOCalculator.calculate(self, atoms, properties, system_changes)

        return
    
    def __check_specorder(self, atoms):
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
    
    def execute(self):
        # check command
        command = self.command

        # run lammps
        proc = subprocess.Popen(
            command, shell=True, cwd=self.directory,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding = 'utf-8'
        )
        errorcode = proc.wait()
        if errorcode: # NOTE: errorcode=1 may be due to lost atoms
            path = os.path.abspath(self.directory)
            msg = ('Failed with command "{}" failed in '
                   '{} with error code {}\n'.format(command, path, errorcode))
            msg += "".join(proc.stdout.readlines())

            raise CalculationFailed(msg)

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
        content += "processors * * 1\n" # if 2D simulation
        
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
        pair_style = self.pair_style
        if pair_style["model"] == "reax/c":
            # reaxff uses real unit, force kcal/mol/A
            content += "pair_style	reax/c NULL\n"
            content += "pair_coeff	* * /users/40247882/projects/oxides/gdp-main/reaxff/ffield.reax.PtO %s\n" %(' '.join(self.specorder))
            content += "neighbor        0.0 bin\n"
            content += "fix             2 all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
            content += "\n"
        elif pair_style["model"] == "eann":
            out_freq = pair_style.get("out_freq", 10)
            if out_freq == 10:
                style_args = "{}".format(pair_style["file"])
            else:
                style_args = "{} out_freq {}".format(pair_style["file"], out_freq)
            content += "pair_style	eann %s\n" %style_args
            content += "pair_coeff	* * double %s\n" %(" ".join(self.specorder))
            content += "neighbor        0.0 bin\n"
            content += "\n"
        elif pair_style["model"] == "deepmd":
            content += "pair_style	deepmd %s\n" %pair_style["file"]
            content += "pair_coeff	\n" 
            content += "neighbor        0.0 bin\n"
            content += "\n"

        # constraint
        constraint = self.parameters["constraint"]
        if constraint is not None:
            # content += "region bottom block INF INF INF INF 0.0 %f\n" %zmin # unit A
            content += "group frozen id %s\n" %constraint
            content += "fix 1 frozen setforce 0.0 0.0 0.0\n"
            content += "\n"

        # outputs
        content += "thermo_style    custom step pe ke etotal temp press vol fmax fnorm\n"
        content += "thermo          10\n"
        content += "dump		1 all custom 10 surface.dump id type x y z fx fy fz\n"
        content += "\n"
        
        # minimisation
        content += "min_style       fire\n"
        content += "min_modify      integrator verlet tmax 4 # see more on lammps doc about min_modify\n"
        content += "minimize        0.0 %f %d %d # energy tol, force tol, step, force step\n" %(
            self.parameters["fmax"] / self.CONVERTOR[self.units], 
            self.parameters["steps"], 2.0*self.parameters["steps"]
        )

        in_file = os.path.join(self.directory, 'in.lammps')
        with open(in_file, 'w') as fopen:
            fopen.write(content)

        return
 

if __name__ == '__main__':
    # test
    calc = Lammps(
        command = "lmp_cat -in ./in.lammps 2>&1 > lmp.out",
        directory =  "./LmpMin-worker",
        pair_style = {
            "file": "/mnt/scratch2/users/40247882/catsign/eann-main/m01r/ensemble/model-0/eann_best_lmp_DOUBLE.pt",
            "model": "eann"
        }
    )

    atoms = read("/mnt/scratch2/users/40247882/catsign/eann-main/m01r/ga-surface/cand2.xyz")
    atoms.calc = calc
    print(atoms.get_potential_energy())

    worker = LmpDynamics(calc, directory=calc.directory)
    min_atoms, min_results = worker.minimise(atoms, fmax=0.2, steps=100, constraint="0:12 24:36")
    print(min_atoms)
    print(min_results)
