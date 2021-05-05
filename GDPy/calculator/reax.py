"""An ASE calculator interface.

"""

import os
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


class ReaxLMP(Calculator):
    name = "ReaxLMP"
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, command=None, label=name, type_dict=None, **kwargs):
        Calculator.__init__(self, command=command, label=label, **kwargs)

        self.command = command

        return
    
    def calculate(self, atoms=None, properties=['energy'],
            system_changes=all_changes):
        # init for creating the directory
        Calculator.calculate(self, atoms, properties, system_changes)

        # elements
        specorder = list(set(atoms.get_chemical_symbols()))
        specorder.sort() # by alphabet

        # write input
        stru_data = os.path.join(self.directory, 'stru.data')
        write_lammps_data(
            stru_data, self.atoms, specorder=specorder, force_skew=True, units='real', atom_style='charge'
        )
        self.write_reax_in(specorder)

        self.run_lammps()

        # obtain results
        self.results = {}

        # read energy
        with open(os.path.join(self.directory, 'log.lammps'), 'r') as fopen:
            lines = fopen.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('Minimization stats:'):
                stat_idx = idx
                break
        else:
            raise ValueError('error in lammps minimization.')
        self.results['energy'] = float(lines[stat_idx+3].split()[-1]) * units.kcal / units.mol

        # read forces from dump file
        dump_atoms = read(
            os.path.join(self.directory, 'surface.dump'), ':', 'lammps-dump-text', 
            specorder=specorder, units='real'
        )[-1]
        self.results['forces'] = dump_atoms.get_forces()

        return
    
    def run_lammps(self):
        # calculate
        command = (
            "%s " %self.command + "-in in.lammps 2>&1 > lmp.out"
        )

        # run lammps
        proc = subprocess.Popen(command, shell=True, cwd=self.directory)
        errorcode = proc.wait()
        if errorcode:
            path = os.path.abspath(self.directory)
            msg = ('Failed with command "{}" failed in '
                   '{} with error code {}'.format(command, path, errorcode))

            raise ValueError(msg)

        return 

    def write_reax_in(self, specorder):
        """"""
        mass_line = ''.join(
            'mass %d %f\n' %(idx+1,atomic_masses[atomic_numbers[elem]]) for idx, elem in enumerate(specorder)
        )
        # write in.lammps
        content = "units           real\n"
        content += "boundary        p p p\n"
        content += "atom_style      charge\n"
        content += "\n"
        content += "neighbor        1.0 bin\n"
        content += "\n"
        content += "box             tilt large\n"
        content += "read_data	    stru.data\n"
        content += "change_box      all triclinic\n"
        content += mass_line
        content += "\n"
        content += "pair_style	reax/c NULL\n"
        content += "pair_coeff	* * /users/40247882/projects/oxides/gdp-main/reaxff/ffield.reax.PtO %s\n" %(' '.join(specorder))
        content += "fix             2 all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
        content += "\n"
        if False:
            content += "region bottom block INF INF INF INF 0.0 4.5\n"
            content += "group bottom_layer region bottom\n"
            content += "fix 1 bottom_layer setforce 0.0 0.0 0.0\n"
            content += "\n"
        content += "thermo_style    custom step pe ke etotal temp press vol fmax fnorm\n"
        content += "thermo          10\n"
        content += "\n"
        content += "dump		1 all custom 10 surface.dump id type x y z fx fy fz\n"
        content += "\n"
        content += "min_style       fire\n"
        content += "min_modify      integrator verlet tmax 4 # see more on lammps doc about min_modify\n"
        content += "minimize        0.0 0.05 0 1000 # energy tol, force tol, step, force step\n"

        in_file = os.path.join(self.directory, 'in.lammps')
        with open(in_file, 'w') as fopen:
            fopen.write(content)

        return
 

if __name__ == '__main__':
    # reaxff uses real unit, force kcal/mol/A
    from ase import Atom, Atoms
    calc = ReaxLMP(
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

