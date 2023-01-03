#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import pathlib
from pathlib import Path
from typing import NoReturn, List, Union

import numpy as np

import matplotlib
matplotlib.use('Agg') #silent mode
import matplotlib.pyplot as plt
#plt.style.use("presentation")

from ase import Atoms
from ase.io import read, write

from GDPy.potential.register import PotentialRegister
from GDPy.utils.command import parse_input_file

from GDPy.validator.validator import AbstractValidator

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

    """
    calculate energies on each structures and save them to file
    """

    def __init__(self, task_outpath: str, task_params: dict, pot_manager=None):
        """
        """
        self.task_outpath = Path(task_outpath)
        self.task_params = task_params
        self.pm = pot_manager

        self.calc = pot_manager.calc
        
        self.structure_paths = self.task_params.get("structures", None)

        return

    def run(self):
        """
        lattice constant
        equation of state
        """
        for stru_path in self.structure_paths:
            # set output file name
            stru_path = Path(stru_path)
            stru_name = stru_path.stem
            fname = self.task_outpath / (stru_name + "-valid.dat")
            pname = self.task_outpath / (stru_name + "-valid.png")

            print(stru_path)

            # run dp calculation
            frames = read(stru_path, ":")
            natoms_array = [len(a) for a in frames]
            volumes = [a.get_volume() for a in frames]
            dft_energies = [a.get_potential_energy() for a in frames]

            mlp_energies = []
            self.calc.reset()
            for a in frames:
                a.calc = self.calc
                mlp_energies.append(a.get_potential_energy())

            # save to data file
            data = np.array([natoms_array, volumes, dft_energies, mlp_energies]).T
            np.savetxt(fname, data, fmt="%12.4f", header="natoms Prop DFT MLP")

            self.plot_dimer(
                "Bulk EOS", volumes, 
                {
                    "DFT": dft_energies, 
                    "MLP": mlp_energies
                },
                pname
            )

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

    def analyse(self):        
        # plot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,12))
        plt.suptitle(
            #"Birch-Murnaghan (Constant-Volume Optimisation)"
            "Energy-Volume Curve"
        )
    
        ax.set_xlabel("Volume [Å^3/atom]")
        ax.set_ylabel("Energy [eV/atom]")

        ax.scatter(volumes/natoms_array, dft_energies/natoms_array, marker="*", label="DFT")
        ax.scatter(volumes/natoms_array, mlp_energies/natoms_array, marker="x", label="MLP")

        ax.legend()

        plt.savefig('bm.png')
        """

        return

def run_validation(
    directory: Union[str, pathlib.Path], input_json: Union[str, pathlib.Path],
    pot_manager
):
    """ This is a factory to deal with various validations...
    """
    # parse potential
    pm = pot_manager

    # run over validations
    valid_dict = parse_input_file(input_json)

    wdir = pathlib.Path(directory)
    if wdir.exists():
        warnings.warn("Validation wdir exists.", UserWarning)

    tasks = valid_dict.get("tasks", {})
    if len(tasks) == 0:
        raise RuntimeError(f"No tasks was found in {input_json}")
    
    for task_name, task_params in tasks.items():
        print(f"=== Run Validation Task {task_name} ===")
        task_outpath = wdir/task_name
        if not task_outpath.exists():
            task_outpath.mkdir(parents=True)
        method = task_params.get("method", "minima")
        # test surface related energies
        if method == "dimer":
            from GDPy.validator.dimer import DimerValidator
            rv = DimerValidator(task_outpath, task_params, pm)
        elif method == "rdf":
            from GDPy.validator.rdf import RdfValidator
            rv = RdfValidator(task_outpath, task_params, pm)
        elif method == "minima":
            from GDPy.validator.minima import MinimaValidator
            rv = MinimaValidator(task_outpath, task_params, pm)
        elif method == "rxn":
            from GDPy.validator.rxn import ReactionValidator
            rv = ReactionValidator(task_outpath, task_params, pm)
        elif method == "bulk":
            rv = SinglePointValidator(task_outpath, task_params, pm)
        rv.run()
        #rv.analyse()

    return


if __name__ == "__main__":
    pass
