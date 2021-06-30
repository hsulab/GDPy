#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np

from ase.io import read, write

from GDPy.utils.command import run_command

class PotentialManager():

    def __init__(self, ensemble_path):

        self.ensemble_path = pathlib.Path(ensemble_path)

        return
    
    def prepare(self):
        """ prepare dataset and training input dirs
        """

        return
    
    def create(self):
        """ create
        """
        return
    
    def freeze(self):
        """ freeze trained potentials
        """
        # find models
        model_dirs = []
        for p in self.ensemble_path.glob('model*'):
            model_dirs.append(p)
        model_dirs.sort()

        # freeze models
        for model_dir in model_dirs:
            if self.check_finished(model_dir):
                self.freeze_model(model_dir)
            self.freeze_model(model_dir)

        return

    @staticmethod
    def check_finished(model_path):
        """check if the training is finished"""
        converged = False
        model_path = pathlib.Path(model_path)
        dpout_path = model_path / 'dp.out'
        if dpout_path.exists():
            content = dpout_path.read_text()
            line = content.split('\n')[-3]
            print(line)
            #if 'finished' in line:
            #    converged = True

        return converged

    @staticmethod
    def freeze_model(model_path):
        command = 'dp freeze -o graph.pb'
        output = run_command(model_path, command)
        print(output)
        return 


if __name__ == '__main__':
    ensemble_path = pathlib.Path('/users/40247882/projects/oxides/gdp-main/it-0009/ensemble')
    pm = PotentialManager(ensemble_path)
    pm.freeze()
    pass
