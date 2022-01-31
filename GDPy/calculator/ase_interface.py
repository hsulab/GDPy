#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import time
import json
import pathlib

import numpy as np

from ase import units
from ase.data import atomic_numbers, atomic_masses
from ase.io import read, write
from ase.build import make_supercell
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms

from ase.optimize import BFGS

from ase.calculators.emt import EMT

from GDPy.md.md_utils import force_temperature

#from GDPy.calculator.dp import DP

from GDPy.md.nosehoover import NoseHoover

class AseDynamics():

    def __init__(self, calc=None, directory="./", logfile="dyn.log", trajfile="dyn.traj"):

        self.calc = calc
        self.calc.reset()

        self.logfile = logfile
        self.trajfile = trajfile
        self.set_output_path(directory)

        self.dynamics = BFGS

        return
    
    def reset(self):
        """ remove calculated quantities
        """
        self.calc.reset()

        return
    
    def set_output_path(self, directory):
        """"""
        self._directory_path = pathlib.Path(directory)
        self._logfile_path = self._directory_path / self.logfile
        self._trajfile_path = self._directory_path / self.trajfile

        return
    
    def run(self, atoms, **kwargs):
        """"""
        # calc_old = atoms.calc
        # params_old = copy.deepcopy(self.calc.parameters)

        # set special keywords
        atoms.calc = self.calc

        if not self._directory_path.exists():
            self._directory_path.mkdir(parents=True)

        # run dynamics
        dyn = self.dynamics(
            atoms, logfile=self._logfile_path,
            trajectory=str(self._trajfile_path)
        )
        dyn.run(kwargs["fmax"], kwargs["steps"])

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
        with open(self._logfile_path, "r") as fopen:
            min_results = fopen.read()

        return min_atoms, min_results

class AseInput():

    def __init__(self, type_map, data, potential, variables, constraint=None):
        """"""
        # structure data
        self.type_map = type_map
        self.data = data

        # potential
        self.potential = potential

        # Be careful with units. 
        # dp uses metal while reax uses real
        self.nsteps = variables.get('nsteps', 1000)
        self.thermo_freq = variables.get('thermo_freq', 10)
        self.dtime = variables.get('dtime', 2) # unit ps in metal 
        self.temp = variables.get('temp', 300)
        self.pres = variables.get('pres', -1)
        self.tau_t = variables.get('tau_t', 0.1) # unit ps
        self.tau_p = variables.get('tau_p', 0.5) # ps

        # constraint
        if constraint is None:
            constraint = [None, None]
        self.constraint = constraint

        return
    
    def update_params(self):
        self.input_dict = dict(
            type_map = self.type_map,
            data = str(self.data),
            potential = self.potential, 
            nsteps = self.nsteps,
            timestep = self.dtime,
            temperature = self.temp,
            # pressure = self.pres,
            disp_freq = self.thermo_freq,
            constrained_indices = self.constraint
        )

        return
    
    def write(self, dir_path, fname='ase.json'):
        """write the input"""
        self.update_params()
        input_path = os.path.join(dir_path, fname)
        with open(input_path, 'w') as fopen:
            json.dump(self.input_dict, fopen, indent=4)
        return


def write_model_devi(fname, step, atoms):
    """"""
    # DP
    #energies_stdvar = atoms.calc.results.get('energies_stdvar', None)
    #forces_stdvar = atoms.calc.results.get('forces_stdvar', None)
    # EANN
    energy_stdvar = atoms.calc.results.get("en_stdvar", None)
    energies_stdvar = atoms.calc.results.get('energies_stdvar', [0.])
    forces_stdvar = atoms.calc.results.get("force_stdvar", None)
    #print(energy_stdvar)
    #print(forces_stdvar)
    content = "{:>12d}" + " {:>18.6e}"*7 + "\n"
    content = content.format(
        step, energy_stdvar,
        np.max(energies_stdvar), np.min(energies_stdvar), np.mean(energies_stdvar),
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
    

def run_ase_calculator(input_json, pot_json):
    """"""
    # parse calculator
    from ..potential.manager import create_manager
    pm = create_manager(pot_json)

    # 
    with open(input_json, "r") as fopen:
        input_dict = json.load(fopen)
    
    temperature = input_dict['temperature']
    
    print('===== temperature at %.2f =====' %temperature)
    # read potential
    calc = pm.generate_calculator()

    # read structure
    type_map = input_dict["type_map"]
    data_path = input_dict['data']
    z_types = [atomic_numbers[x] for x in type_map.keys()]
    atoms = read(data_path, format='lammps-data', style='atomic', units='metal', Z_of_type=z_types)

    # set calculator and constraints
    constrained_string = input_dict['constrained_indices'][1] # 0-flexible 1-frozen
    if constrained_string is not None:
        con_indices = []
        for con_string in constrained_string.split():
            start, end = con_string.split(':')
            con_indices.extend(list(range(int(start)-1,int(end))))
        cons = FixAtoms(indices=con_indices)
        atoms.set_constraint(cons)
    atoms.calc = calc

    # run MD
    MaxwellBoltzmannDistribution(atoms, temperature*units.kB)
    force_temperature(atoms, temperature)

    timestep = input_dict["timestep"]

    nvt_dyn = NoseHoover(
        atoms = atoms,
        timestep = timestep * units.fs,
        temperature = temperature * units.kB,
        nvt_q = 334.
    )

    cwd = pathlib.Path(os.getcwd())
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
        content = "{:>12s}" + " {:>18s}"*7 + "\n"
        content = content.format(
            '#       step', 'tot_devi_e',
            'max_devi_e', 'min_devi_e', 'avg_devi_e',
            'max_devi_f', 'min_devi_f', 'avg_devi_f'
        )
        fopen.write(content)

    # calculate at the first step
    #atoms.calc.calc_uncertainty = True
    #dummy = atoms.get_forces()

    st = time.time()

    nsteps = input_dict['nsteps']
    check_stdvar_freq = input_dict['disp_freq']
    for step in range(-1,nsteps+check_stdvar_freq):
        if step % check_stdvar_freq == 0:
            atoms.calc.calc_uncertainty = True
            nvt_dyn.step()
            write(xyz_fname, atoms, append=True)
            write_md_info(out_fname, step, atoms)
            write_model_devi(devi_fname, step, atoms)
        else:
            atoms.calc.calc_uncertainty = False
            nvt_dyn.step()

    et = time.time()

    print("time cost: ", et-st)

    return


if __name__ == '__main__':
    pass