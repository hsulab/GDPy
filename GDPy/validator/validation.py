#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from GDPy.calculator.dp import DP

from abc import ABC
from abc import abstractmethod


class AbstractValidator(ABC):

    def __init__(self, validation: Union[str, pathlib.Path], *args, **kwargs):
        """"""
        with open(validation, 'r') as fopen:
            valid_dict = json.load(fopen)

        self.tasks = valid_dict['tasks']
        self.output_path = pathlib.Path(valid_dict['output'])
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)

        self.calc = DP(
            type_dict = valid_dict['potential']['type_map'],
            model = valid_dict['potential']['model']
        )

        return

    @abstractmethod
    def run(self, *args, **kwargs):
        return


class ReactionValidator(AbstractValidator):

    def __init__(self, validation: Union[str, pathlib.Path]):
        """ reaction formula
            how to postprocess
        """
        super().__init__(validation)

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
    
    def run(self):
        self.outputs = []
        for (task_name, task_data) in self.tasks.items():
            output_data = {'basics': [], 'composites': []}
            count = 0
            cons_data = task_data['constraints']
            assert len(cons_data) == len(task_data['basics']) + len(task_data['composites'])
            dynamics = getattr(ase.optimize, task_data['dynamics'][0])
            for coef, stru_file in task_data['basics']:
                atoms_name = pathlib.Path(stru_file).stem
                atoms = read(stru_file)
                atoms.calc = self.calc
                if cons_data[count] is not None:
                    cons = FixAtoms(
                        indices = [atom.index for atom in atoms if atom.z < cons_data[count]]
                    )
                    atoms.set_constraint(cons)
                dyn = dynamics(atoms)
                dyn.run(task_data['dynamics'][1], task_data['dynamics'][2])
                output_data['basics'].append([atoms_name, coef, atoms])
                count += 1
            for coef, stru_file in task_data['composites']:
                atoms_name = pathlib.Path(stru_file).stem
                atoms = read(stru_file)
                atoms.calc = self.calc
                if cons_data[count] is not None:
                    cons = FixAtoms(
                        indices = [atom.index for atom in atoms if atom.z < cons_data[count]]
                    )
                    atoms.set_constraint(cons)
                dyn = dynamics(atoms)
                dyn.run(task_data['dynamics'][1], task_data['dynamics'][2])
                output_data['composites'].append([atoms_name, coef, atoms])
                count += 1
            self.outputs.append(output_data)

        return
    
    def analyse(self):
        saved_frames = []
        for output_data in self.outputs:
            basics_data = output_data['basics']
            composites_data = output_data['composites']
            for atoms_name, coefs, atoms in composites_data:
                assert len(coefs) == len(basics_data)
                relative_energy = atoms.get_potential_energy()
                for idx, coef in enumerate(coefs):
                    relative_energy -= coef*basics_data[idx][1]*basics_data[idx][2].get_potential_energy()
                saved_frames.append(atoms)
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
        with open(validation, 'r') as fopen:
            valid_dict = json.load(fopen)

        self.tasks = valid_dict['tasks']
        self.output = pathlib.Path(valid_dict['output'])
        if not self.output.exists():
            self.output.mkdir(parents=True)

        self.calc = DP(
            type_dict = valid_dict['potential']['type_map'],
            model = valid_dict['potential']['model']
        )

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
            output_path = self.output
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




if __name__ == '__main__':
    # sg = SinglePointValidator('./valid.json')
    # sg.run()

    # test surface related energies
    rv = ReactionValidator('valid-opt3.json')
    rv.run()
    rv.analyse()