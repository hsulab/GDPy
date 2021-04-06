#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run molecular dynamics (NVE/NVT/NPT) to explore configuration space
"""

import numpy as np 

from pathlib import Path

from ase import units
from ase.io import read, write
from ase.build import make_supercell
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

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


def loop_over_temperatures(cwd, atoms, temperatures, model_path):
    """run MD at various temperatures"""
    calc = DP(
        model=model_path,
        type_dict={'O': 0, 'Pt': 1}
    )

    for temp in temperatures:
        md_atoms = atoms.copy()
        md_atoms.calc = calc
        md_atoms.calc.reset()
        run_md(cwd, md_atoms, temp, nsteps=500, save_traj=True)

    return 

def run_md(cwd, atoms, temperature, nsteps=500, save_traj=False):
    print('===== temperature at %.2f =====' %temperature)
    # run MD
    MaxwellBoltzmannDistribution(atoms, temperature*units.kB)

    timestep = 2.0

    nvt_dyn = NoseHoover(
        atoms = atoms,
        timestep = timestep * units.fs,
        temperature = temperature * units.kB,
        nvt_q = 334.
    )

    #nvt_dyn.attach(print_temperature, atoms=atoms)
    #nvt_dyn.run(steps=10)
    xyz_fname = 'temp-'+str(temperature)+'K.xyz'
    with open(test_dir/xyz_fname, 'w') as fopen:
        fopen.write('')

    traj_xyz_fname = 'traj-'+str(temperature)+'K.xyz'
    with open(test_dir/traj_xyz_fname, 'w') as fopen:
        fopen.write('')

    check_stdvar_freq = 10
    for step in range(nsteps):
        #print('old', atoms.positions.flatten())
        nvt_dyn.step()
        #print('new', atoms.positions.flatten())
        if step % check_stdvar_freq == 0:
            write(test_dir/traj_xyz_fname, atoms, append=True)
            unlearned, max_stdvar = check_stdvar(atoms)
            if unlearned:
                write(test_dir/xyz_fname, atoms, append=True)
                print('unlearned!!!')
            print('step %d with max_stdvar %.4f' %(step, max_stdvar))

        pass

def sample_configuration(iter_directory, main_database):
    """
    explore configurations with a given system (fixed box and fixed number of elements)
    """
    # read model ensemble
    ensemble_path = iter_directory / 'ensemble'

    model_dirs = []
    for p in ensemble_path.glob('model*'):
        model_dirs.append(p)
    model_dirs.sort()

    graph_ensemble = [str(model/'graph.pb') for model in model_dirs]

    # setting MD parameters
    test_dir = "/users/40247882/projects/oxides/dptrain"
    test_dir = Path(test_dir)

    init_stru = 'opts/Pt3O4_opt.xyz' # initial structure path

    atoms = read(test_dir/init_stru)
    atoms = make_supercell(atoms, 2.0*np.eye(3)) # (2x2x2) cell
    print(atoms.cell)

    temperatures = [300, 600, 1200, 2400]

    loop_over_temperatures(test_dir, atoms, temperatures, graph_ensemble)

    return 
    

if __name__ == '__main__':
    """"""
    pass
