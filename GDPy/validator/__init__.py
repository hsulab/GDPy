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
        elif method == "singlepoint":
            from GDPy.validator.singlepoint import SinglepointValidator
            rv = SinglepointValidator(task_outpath, task_params, pm)
        elif method == "minima":
            from GDPy.validator.minima import MinimaValidator
            rv = MinimaValidator(task_outpath, task_params, pm)
        elif method == "traj":
            from GDPy.validator.traj import TrajValidator
            rv = TrajValidator(task_outpath, task_params, pm)
        elif method == "rxn":
            from GDPy.validator.rxn import ReactionValidator
            rv = ReactionValidator(task_outpath, task_params, pm)
        else:
            raise NotImplementedError(f"Validation {method} is not supported.")
        rv.run()
        #rv.analyse()

    return


if __name__ == "__main__":
    pass
