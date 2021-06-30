#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.potential.manager import PotManager
import json
import pathlib
from typing import Union

from collections import namedtuple, Counter
from ase.io.trajectory import OldCalculatorWrapper

import numpy as np

import matplotlib
matplotlib.use('Agg') #silent mode
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write

import ase.optimize
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.constraints import UnitCellFilter

from GDPy.potential.manager import PotManager

from abc import ABC
from abc import abstractmethod

"""
Various properties to be validated

Atomic Energy and Crystal Lattice constant

Elastic Constants

Phonon Calculations

Point Defects (vacancies, self interstitials, ...)

Surface energies

Diffusion Coefficient

Adsorption, Reaction, ...
"""


class AbstractValidator(ABC):

    def __init__(self, validation: Union[str, pathlib.Path], pot, *args, **kwargs):
        """"""
        with open(validation, 'r') as fopen:
            valid_dict = json.load(fopen)

        self.tasks = valid_dict['tasks']
        self.output_path = pathlib.Path(valid_dict['output'])
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        
        self.pot = pot # potential manager
        self.calc = pot.generate_calculator()

        return

    @abstractmethod
    def run(self, *args, **kwargs):
        return


class ReactionValidator(AbstractValidator):

    def __init__(self, validation: Union[str, pathlib.Path], pot):
        """ reaction formula
            how to postprocess
        """
        super().__init__(validation, pot)

        return

    def run_dynamics(atoms, calc, cons=None):
        if cons is not None:
            atoms.set_constraint(cons)
        atoms.calc = calc
        dyn = BFGS(atoms)
        dyn.run(fmax=0.05, steps=200)

        print('energy: ', atoms.get_potential_energy())
        print(np.linalg.norm(atoms[0].position-atoms[1].position))
        return 
    
    def _run_group(self, group_data: dict, dyn_data):
        """ run group of structures
        """
        group_output = [] # [[name, atoms],...,[]]
        for stru_file, cons_data in zip(group_data['structures'], group_data['constraints']):
            atoms_name = pathlib.Path(stru_file).stem
            print('===== ', atoms_name, ' =====')
            frames = read(stru_file, ':')
            assert len(frames) == 1, 'only one structure at a time now'
            atoms = frames[0]
            atoms.calc = self.calc
            if cons_data[0] == "FixAtoms":
                if cons_data[1] is not None:
                    cons = FixAtoms(
                        indices = [atom.index for atom in atoms if atom.z < cons_data[1]]
                    )
                    atoms.set_constraint(cons)
                    print('constraint: natoms', cons)
                else:
                    pass
            elif cons_data[0] == "UnitCellFilter":
                atoms = UnitCellFilter(atoms, constant_volume=False)
                print('constraint: UnitcellFilter')
            else:
                raise ValueError("unsupported constraint type.")
            dynamics = getattr(ase.optimize, dyn_data[0])
            dyn = dynamics(atoms)
            dyn.run(dyn_data[1], dyn_data[2])
            if self.pot.uncertainty:
                print(atoms.calc.results['energy_stdvar'])
            group_output.append([atoms_name, atoms])

        return group_output
    
    def run(self):
        self.my_references = []
        self.outputs = []
        for (task_name, task_data) in self.tasks.items():
            basics_output = self._run_group(task_data['basics'], task_data['dynamics'])
            composites_output = self._run_group(task_data['composites'], task_data['dynamics'])
            self.outputs.append({'basics': basics_output, 'composites': composites_output})

        return
    
    def analyse(self):
        # check data
        saved_frames = []
        for (task_name, task_data), output_data in zip(self.tasks.items(), self.outputs):
            basics_output = output_data['basics']
            basics_energies = []
            for (atoms_name, atoms), coef in zip(basics_output, task_data['basics']['coefs']):
                basics_energies.append(atoms.get_potential_energy()*coef)
            composites_output = output_data['composites']
            composites_references = task_data['composites'].get('references', None)
            for idx, ((atoms_name, atoms), coef) in enumerate(zip(composites_output, task_data['composites']['coefs'])):
                assert len(basics_energies) == len(coef)
                relative_energy = atoms.get_potential_energy()
                for en, c in zip(basics_energies, coef):
                    relative_energy -= c*en
                saved_frames.append(atoms)
                if composites_references is not None:
                    if self.pot.uncertainty > 1:
                        # print(atoms_name, relative_energy, atoms.info['energy_stdvar'], composites_references[idx])
                        print(atoms_name, relative_energy, composites_references[idx])
                    else:
                        print(atoms_name, relative_energy, composites_references[idx])
                else:
                    print(atoms_name, relative_energy)
        write(self.output_path / 'saved.xyz', saved_frames)

        return


class RunCalculation():

    def __init__(self):

        return 
    
    def run(self, frames, func_name):
        """"""
        func = getattr(self, func_name)
        return func(frames)
    
    @staticmethod
    def dimer(frames):
        """turn xyz into dimer data"""
        data = []
        for atoms in frames:
            # donot consider minimum image
            distance = np.linalg.norm(atoms[0].position-atoms[1].position) 
            energy = atoms.get_potential_energy()
            data.append([distance,energy])
        data = np.array(data)
    
        return np.array(data[:,0]), np.array(data[:,1])

    @staticmethod
    def volume(frames):
        """turn xyz into eos data"""
        data = []
        for atoms in frames:
            # donot consider minimum image
            vol = atoms.get_volume()
            energy = atoms.get_potential_energy()
            data.append([vol,energy])
        data = np.array(data)

        return np.array(data[:,0]), np.array(data[:,1])


class SinglePointValidator(AbstractValidator):

    def __init__(self, validation):
        """"""
        super().__init__()

        return

    def run_calculation(frames, calc):
        dp_energies = []
        for atoms in frames:
            calc.reset()
            atoms.calc = calc
            dp_energies.append(atoms.get_potential_energy())
        dp_energies = np.array(dp_energies)

        return dp_energies
    
    def run(self, calc=None, output_path=None):
        """
        lattice constant
        equation of state
        """
        if output_path is None:
            output_path = self.output_path
        if calc is None:
            calc = self.calc

        runrun = RunCalculation()

        # run over various validations
        for validation, systems in self.tasks.items():
            print(validation, systems)
            for stru_path in systems:
                # set output file name
                stru_path = pathlib.Path(stru_path)
                stru_name = stru_path.stem
                fname = output_path / (stru_name + '-dpx.dat')

                # run dp calculation
                frames = read(stru_path, ':')
                properties, dft_energies = runrun.run(frames, validation)
                dp_energies = self.run_calculation(frames, calc)

                # save to data file
                data = np.array([properties, dft_energies, dp_energies]).T
                np.savetxt(fname, data, fmt='%.4f', header='Prop DFT DP')

                # plot comparison
                # pic_path = output_path / (stru_name+'-dpx.png')
                # print(pic_path)
                # energies = {
                #     'reference': dft_energies, 
                #     'learned': dp_energies
                # }
                # plot_dimer(validation, volumes, energies, pic_path)

        return

    @staticmethod
    def plot_dimer(task_name, distances, energies: dict, pname):
        """"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
        ax.set_title(
            task_name,
            fontsize=20, 
            fontweight='bold'
        )
    
        ax.set_xlabel('Distance [Å]', fontsize=16)
        ax.set_ylabel('Energyr [eV]', fontsize=16)

        for name, en in energies.items():
            ax.scatter(distances, en, label=name)
        ax.legend()

        plt.savefig(pname)

        return

def run_validation(input_json: Union[str, pathlib.Path], pot_params: list):
    # sg = SinglePointValidator('./valid.json')
    # sg.run()
    type_map = {'O': 0, 'Pt': 1}

    # check potential
    pot_name = pot_params[0]
    pot_path = pathlib.Path(pot_params[1])
    pot_dir = pot_path.parent
    pot_pattern = pot_path.name
    print(pot_dir)
    print(pot_pattern)

    models = []
    for pot in pot_dir.glob(pot_pattern):
        models.append(str(pot/'graph.pb'))
    print(models)
    pm = PotManager()
    pot = pm.create_potential(pot_name, 'ase', models, type_map)

    # test surface related energies
    rv = ReactionValidator(input_json, pot)
    rv.run()
    rv.analyse()

    return


if __name__ == '__main__':
    run_validation('./valid-opt.json')