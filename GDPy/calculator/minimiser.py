"""An ASE calculator interface.

"""

from logging import StringTemplateStyle
import os
from posixpath import commonpath
import subprocess

import numpy as np

from ase import units
from ase.data import atomic_numbers, atomic_masses
from ase.io import read, write
from ase.io.lammpsrun import read_lammps_dump_text
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.calculators.calculator import (
    Calculator, all_changes, PropertyNotImplementedError
)
from ase.calculators.singlepoint import SinglePointCalculator


class LMPMin(Calculator):

    name = "LMPMin"
    supported_pairstyles = ["deepmd", "eann", "reax/c"]
    implemented_properties = ["energy", "forces", "stress"]

    STRUCTURE_FILE = "stru.data"

    # only for energy and forces, eV and eV/AA
    CONVERTOR = {
        "metal": 1.0,
        "real": units.kcal / units.mol
    }

    def __init__(
        self, 
        command = None, 
        label = name, 
        type_list: dict = None, 
        units: str = "metal",
        atom_style: str = "atomic",
        model_params = None, # pair_style specific parameters
        **kwargs
    ):
        """"""
        Calculator.__init__(self, command=command, label=label, **kwargs)

        self.command = command
        self.units = units
        self.atom_style = atom_style
        self.model_params = model_params

        return
    
    def calculate(self, atoms=None, properties=['energy'],
            system_changes=all_changes):
        # init for creating the directory
        Calculator.calculate(self, atoms, properties, system_changes)

        # elements
        specorder = list(set(atoms.get_chemical_symbols()))
        specorder.sort() # by alphabet

        # write input
        stru_data = os.path.join(self.directory, self.STRUCTURE_FILE)
        write_lammps_data(
            stru_data, self.atoms, specorder=specorder, force_skew=True, 
            units=self.units, atom_style=self.atom_style
        )
        self.__write_reax_in(specorder)

        self.__run_lammps()

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
            specorder=specorder, units=self.units
        )[-1]
        self.results['forces'] = dump_atoms.get_forces() * self.CONVERTOR[self.units]

        return
    
    def __run_lammps(self):
        # calculate
        #command = (
        #    "%s " %self.command + "-in in.lammps 2>&1 > lmp.out"
        #)
        command = self.command

        # run lammps
        proc = subprocess.Popen(
            command, shell=True, cwd=self.directory,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding = 'utf-8'
        )
        errorcode = proc.wait()
        if errorcode:
            path = os.path.abspath(self.directory)
            msg = ('Failed with command "{}" failed in '
                   '{} with error code {}\n'.format(command, path, errorcode))
            msg += "".join(proc.stdout.readlines())

            raise ValueError(msg)

        return 
    
    def minimise(self, atoms=None, steps=200, fmax=0.05, zmin=None):
        # init for creating the directory
        Calculator.calculate(self, atoms, ['energy'], all_changes)

        # elements in system
        specorder = list(set(atoms.get_chemical_symbols()))
        specorder.sort() # by alphabet

        # write input
        stru_data = os.path.join(self.directory, self.STRUCTURE_FILE)
        write_lammps_data(
            stru_data, self.atoms, specorder=specorder, force_skew=True, 
            units=self.units, atom_style=self.atom_style
        )
        self.__write_reax_in(specorder, steps, fmax, zmin)

        self.__run_lammps()

        # obtain results
        results = {}

        # read energy
        with open(os.path.join(self.directory, 'log.lammps'), 'r') as fopen:
            lines = fopen.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('Minimization stats:'):
                stat_idx = idx
                break
        else:
            raise ValueError('error in lammps minimization.')
        stat_content = ''.join(lines[stat_idx:stat_idx+9])
        results['energy'] = float(lines[stat_idx+3].split()[-1]) * self.CONVERTOR[self.units]

        # read forces from dump file
        dump_atoms = read(
            os.path.join(self.directory, 'surface.dump'), ':', 'lammps-dump-text', 
            specorder=specorder, units=self.units
        )[-1]

        results['forces'] = dump_atoms.get_forces() * self.CONVERTOR[self.units]

        dump_atoms.calc = SinglePointCalculator(
            dump_atoms,
            energy = results["energy"],
            forces = results["forces"]
        )

        return dump_atoms, stat_content
    
    def __write_reax_in(self, specorder, steps=0, fmax=0.05, zmin=None):
        """"""
        mass_line = ''.join(
            'mass %d %f\n' %(idx+1,atomic_masses[atomic_numbers[elem]]) for idx, elem in enumerate(specorder)
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
        if self.model_params["model"] == "reax/c":
            content += "pair_style	reax/c NULL\n"
            content += "pair_coeff	* * /users/40247882/projects/oxides/gdp-main/reaxff/ffield.reax.PtO %s\n" %(' '.join(specorder))
            content += "neighbor        0.0 bin\n"
            content += "fix             2 all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
            content += "\n"
        elif self.model_params["model"] == "eann":
            out_freq = self.model_params.get("out_freq", 10)
            if out_freq == 10:
                style_args = "{}".format(self.model_params["file"])
            else:
                style_args = "{} out_freq {}".format(self.model_params["file"], out_freq)
            content += "pair_style	eann %s\n" %style_args
            content += "pair_coeff	* * double %s\n" %(" ".join(specorder))
            content += "neighbor        0.0 bin\n"
            content += "\n"
        elif self.model_params["model"] == "deepmd":
            content += "pair_style	deepmd %s\n" %self.model_params["file"]
            content += "pair_coeff	\n" 
            content += "neighbor        0.0 bin\n"
            content += "\n"

        # constraint
        if zmin is not None:
            content += "region bottom block INF INF INF INF 0.0 %f\n" %zmin # unit A
            content += "group bottom_layer region bottom\n"
            content += "fix 1 bottom_layer setforce 0.0 0.0 0.0\n"
            content += "\n"

        # outputs
        content += "thermo_style    custom step pe ke etotal temp press vol fmax fnorm\n"
        content += "thermo          10\n"
        content += "dump		1 all custom 10 surface.dump id type x y z fx fy fz\n"
        content += "\n"
        
        # minimisation
        content += "min_style       fire\n"
        content += "min_modify      integrator verlet tmax 4 # see more on lammps doc about min_modify\n"
        content += "minimize        0.0 %f %d %d # energy tol, force tol, step, force step\n" %(fmax/self.CONVERTOR[self.units], steps, 2.0*steps)

        in_file = os.path.join(self.directory, 'in.lammps')
        with open(in_file, 'w') as fopen:
            fopen.write(content)

        return
 

if __name__ == '__main__':
    # reaxff uses real unit, force kcal/mol/A
    from ase import Atom, Atoms
    calc = LMPMin(
        directory = 'reax-worker',
        command='mpirun -n 1 lmp'
    )
    #atoms = Atoms('O2', positions=[[0.,0.,0.],[0.,0.,1.2449]], cell=10.*np.eye(3))
    #atoms.calc = calc
    #print(atoms.get_potential_energy())
    #print(atoms.get_forces())
    #atoms = Atoms('O', positions=[[0.,0.,0.]], cell=10.*np.eye(3))
    atoms = read('./surface-3O.xyz')
    #atoms = read('/users/40247882/projects/oxides/gdp-main/mc-test/Pt_111_0.xyz')
    atoms.calc = calc
    print(atoms.get_potential_energy())
    #print(atoms.get_forces())

