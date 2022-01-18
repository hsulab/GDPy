#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run molecular dynamics (NVE/NVT/NPT) to explore configuration space
"""

import numpy as np 

from pathlib import Path

from ase import units
from ase.data import atomic_numbers, atomic_masses
from ase.io import read, write
from ase.build import make_supercell
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms

from GDPy.md.md_utils import force_temperature

from GDPy.calculator.dp import DP

from GDPy.md.nosehoover import NoseHoover

from abc import ABC
from abc import abstractmethod

class AbstractExplorer(ABC):

    def __init__(self):
        
        return

class MdExplorer(AbstractExplorer):

    def __init__(self, directory, system, machine=None):
        if machine is None:
            self.mode = 'local' # local mode
        else:
            self.mode = 'machine' # submit to machine for large systems

        return

    def add_explorer(self):
        pass

    def explore(self):
        pass


def check_stdvar(atoms):
    """Check the maximum standard variance of current atoms"""
    unlearned = False

    forces_stdvar = atoms.calc.results.get('forces_stdvar', None)
    if forces_stdvar is not None:
        max_stdvar = np.max(forces_stdvar)
        if max_stdvar < 0.05:
            pass # small acceptable error
        elif max_stdvar < 0.20:
            unlearned = True 
        else:
            pass # very large error 
    else:
        max_stdvar = 1e6
    
    return unlearned, max_stdvar

def write_model_devi(fname, step, atoms):
    """"""
    energy_stdvar = atoms.calc.results.get('energy_stdvar', None)
    forces_stdvar = atoms.calc.results.get('forces_stdvar', None)
    content = "{:>12d}" + " {:>18.6e}"*6 + "\n"
    content = content.format(
        step, np.max(energy_stdvar), np.min(energy_stdvar), np.mean(energy_stdvar),
        np.max(forces_stdvar), np.min(forces_stdvar), np.mean(forces_stdvar)
    )
    with open(fname, 'a') as fopen:
        fopen.write(content)

    return

def write_md_info(fname, step, atoms):
    content = "{:>12d}" + " {:>18.6f}"*2 + "\n"
    content = content.format(
        step, atoms.get_temperature(), atoms.get_potential_energy()
    )
    with open(fname, 'a') as fopen:
        fopen.write(content)

    return 


def run_md(cwd, atoms, temperature, nsteps=500):
    print('===== temperature at %.2f =====' %temperature)
    # run MD
    MaxwellBoltzmannDistribution(atoms, temperature*units.kB)
    force_temperature(atoms, temperature)

    timestep = 2.0

    nvt_dyn = NoseHoover(
        atoms = atoms,
        timestep = timestep * units.fs,
        temperature = temperature * units.kB,
        nvt_q = 334.
    )

    #nvt_dyn.attach(print_temperature, atoms=atoms)
    #nvt_dyn.run(steps=10)
    xyz_fname = cwd / 'traj.xyz'
    with open(xyz_fname, 'w') as fopen:
        fopen.write('')

    out_fname = cwd / 'ase.out'
    with open(out_fname, 'w') as fopen:
        content = "{:>12s}" + " {:>18s}"*2 + "\n"
        content = content.format(
            '#       step', 'temperature', 'pot energy'
        )
        fopen.write(content)

    devi_fname = cwd / 'model_devi.out'
    with open(devi_fname, 'w') as fopen:
        content = "{:>12s}" + " {:>18s}"*6 + "\n"
        content = content.format(
            '#       step',
            'max_devi_e', 'min_devi_e', 'avg_devi_e',
            'max_devi_f', 'min_devi_f', 'avg_devi_f'
        )
        fopen.write(content)

    dummy = atoms.get_forces()
    check_stdvar_freq = 10
    for step in range(nsteps):
        if step % check_stdvar_freq == 0:
            write(xyz_fname, atoms, append=True)
            write_md_info(out_fname, step, atoms)
            write_model_devi(devi_fname, step, atoms)
        nvt_dyn.step()
    
    return


def sample_configuration(data_path, type_map, model, sample_variables, temperatures):
    """
    explore configurations with a given system (fixed box and fixed number of elements)
    """
    z_types = [atomic_numbers[x] for x in type_map.keys()]
    atoms = read(data_path, format='lammps-data', style='atomic', units='metal', Z_of_type=z_types)

    if True:
        height = 5.0
        indices = [atom.index for atom in atoms if atom.position[2] < height]
        atoms.set_constraint(FixAtoms(indices=indices))

    calc = DP(
        model=model,
        type_dict=type_map
    )
    print(sample_variables)
    nsteps = sample_variables['nsteps']+1 # this can save the last step

    name_path = Path('/users/40247882/projects/oxides/gdp-main/it-0003/MD0/Pt111-nvt')
    for temp in temperatures:
        temp_dir = name_path / str(temp)
        try:
            temp_dir.mkdir(parents=True)
        except FileExistsError:
            print('skip this %s' %temp_dir)
            continue
        md_atoms = atoms.copy()
        md_atoms.calc = calc
        md_atoms.calc.reset()
        run_md(temp_dir, md_atoms, temp, nsteps=nsteps)

    return 


if __name__ == '__main__':
    """"""
    data_path = '/users/40247882/projects/oxides/gdp-main/init-systems/Pt111.data'
    type_map = {'O': 0, 'Pt': 1}
    model = [
        '/users/40247882/projects/oxides/gdp-main/it-0003/ensemble/model-0/graph.pb', 
        '/users/40247882/projects/oxides/gdp-main/it-0003/ensemble/model-1/graph.pb', 
        '/users/40247882/projects/oxides/gdp-main/it-0003/ensemble/model-2/graph.pb', 
        '/users/40247882/projects/oxides/gdp-main/it-0003/ensemble/model-3/graph.pb'
    ]
    sample_variables = {
        'nsteps': 1000, 
        'thermo_freq': 10, 
        'dtime': 0.002, 
        'temp': 300, 
        'pres': -1, 
        'tau_t': 0.1, 
        'tau_p': 0.5
    }
    temperatures = [150, 300, 450, 600, 900, 1200, 1500, 1800, 2100, 2400]

    sample_configuration(data_path, type_map, model, sample_variables, temperatures)
    pass
