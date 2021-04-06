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

    def __init__(self, directory, machine):
        pass

    def add_explorer(self):
        pass


def check_stdvar(atoms):
    #content = 'temperature %8.4f' %atoms.get_temperature()
    #print(content)
    unlearned = False

    forces_stdvar = atoms.calc.results.get('forces_stdvar', None)
    if forces_stdvar is not None:
        max_stdvar = np.max(forces_stdvar)
        if max_stdvar < 0.05:
            pass # small acceptable error
        elif max_stdvar < 0.30: # 0.20 too small for oxides
            unlearned = True 
        else:
            pass # very large error 
    else:
        max_stdvar = 1e6
    
    return unlearned, max_stdvar

md_params = {
    'temperature': 300, # K
    'timestep': 2.0 # fs
}


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
        run_md(cwd, md_atoms, temp, nsteps=500)

    return 

def run_md(cwd, atoms, temperature, nsteps=500):
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

    check_stdvar_freq = 10
    for step in range(nsteps):
        #print('old', atoms.positions.flatten())
        nvt_dyn.step()
        #print('new', atoms.positions.flatten())
        if step % check_stdvar_freq == 0:
            unlearned, max_stdvar = check_stdvar(atoms)
            if unlearned:
                write(test_dir/xyz_fname, atoms, append=True)
                print('!!!')
            print('step %d with max_stdvar %.4f' %(step, max_stdvar))

        pass


if __name__ == '__main__':
    """"""
    # read model ensemble
    ensemble_path = Path('/users/40247882/projects/oxides/dptrain/ensemble-2')
    model_dirs = []
    for p in ensemble_path.glob('model*'):
        model_dirs.append(p)
    model_dirs.sort()

    graph_ensemble = [str(model/'graph.pb') for model in model_dirs]
    print(graph_ensemble)

    # setting MD parameters
    print('===== GDPy =====')
    test_dir = "/users/40247882/projects/oxides/dptrain"
    test_dir = Path(test_dir)

    init_stru = 'opts/Pt3O4_opt.xyz' # initial structure path

    atoms = read(test_dir/init_stru)
    atoms = make_supercell(atoms, 2.0*np.eye(3))
    print(atoms.cell)

    temperatures = [300, 600, 1200, 2400]

    loop_over_temperatures(test_dir, atoms, temperatures, graph_ensemble)
    pass
