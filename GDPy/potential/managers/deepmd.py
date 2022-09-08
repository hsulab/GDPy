#!/usr/bin/env python3
# -*- coding: utf-8 -*

from pathlib import Path
from typing import Union

import json

from GDPy.potential.potential import AbstractPotential
from GDPy.utils.command import run_command
from GDPy.trainer.train_potential import find_systems, generate_random_seed


class DeepmdManager(AbstractPotential):

    name = "deepmd"
    implemented_backends = ['ase', 'lammps']

    def __init__(self, backend: str, models: Union[str, list], type_map: dict):
        """ create a dp manager
        """
        self.backend = backend
        if self.backend not in self.implemented_backends:
            raise NotImplementedError('Backend %s is not implemented.' %self.backend)

        # check models
        self.models = models
        self._parse_models()
        self._check_uncertainty_support()

        self.type_map = type_map

        return
    
    def _parse_models(self):
        """"""
        if isinstance(self.models, str):
            pot_path = Path(self.models)
            pot_dir, pot_pattern = pot_path.parent, pot_path.name
            models = []
            for pot in pot_dir.glob(pot_pattern):
                models.append(str(pot/'graph.pb'))
            self.models = models
        else:
            for m in self.models:
                if not Path(m).exists():
                    raise ValueError('Model %s does not exist.' %m)

        return
    
    def _check_uncertainty_support(self):
        """"""
        self.uncertainty = False
        if len(self.models) > 1:
            self.uncertainty = True

        return
    
    def register_calculator(self):
        """ generate calculator with various backends
        """
        if self.backend == 'ase':
            # return ase calculator
            from GDPy.computation.dp import DP
            calc = DP(model=self.models, type_dict=self.type_map)
        elif self.backend == 'lammps':
            # return deepmd pair related content
            #content = "units           metal\n"
            #content += "atom_style      atomic\n"
            content = "neighbor        1.0 bin\n"
            content += "pair_style      deepmd %s out_freq ${THERMO_FREQ} out_file model_devi.out\n" \
                %(' '.join([m for m in self.models]))
            content += 'pair_coeff    \n'
            calc = content

        return calc
    
    def create_thermostat(self):

        return
    
    def create_ensemble(self):
        """
        """
        # TODO: make these paramters
        main_dict = []
        iter_directory = ""
        find_systems = ""
        generate_random_seed = ""

        # parse params
        machine_json = main_dict['machines']['trainer']
        num_models = main_dict['training']['num_models']
        train_json = main_dict['training']['json']

        # prepare dataset
        """
        xyz_files = []
        for p in main_database.glob('*.xyz'):
            xyz_files.append(p)
    
        frames = []
        for xyz_file in xyz_files:
            frames.extend(read(xyz_file, ':'))
    
        print(len(frames))
        """

        # find systems
        # data_path = Path('/users/40247882/projects/oxides/gdp-main/merged-dataset/raw_data')
        data_path = iter_directory / 'raw_data'
        systems = find_systems(data_path)

        # machine file
        from GDPy.scheduler.scheduler import SlurmScheduler
        slurm_machine = SlurmScheduler(machine_json)

        # read json
        with open(train_json, 'r') as fopen:
            params_dict = json.load(fopen)
        # change seed and system dirs
        ensemble_dir = iter_directory / 'ensemble'
        ensemble_dir.mkdir()
        for idx in range(num_models):
            #print(params_dict)
            model_dir = ensemble_dir / ('model-'+str(idx))
            model_dir.mkdir()
            seed = generate_random_seed()
            params_dict['model']['descriptor']['seed'] = int(seed)
            params_dict['model']['fitting_net']['seed'] = int(seed)

            params_dict['training']['systems'] = [str(s) for s in systems]

            with open(model_dir/'dp.json', 'w') as fopen:
                json.dump(params_dict, fopen, indent=4)

            # write machine 
            #restart = True
            restart = False
            slurm_machine.machine_dict['job-name'] = 'model-'+str(idx)
            if restart:
                parent_model = '/users/40247882/projects/oxides/gdp-main/it-0008/ensemble/model-%d/model.ckpt' %idx
                command = "dp train ./dp.json --init-model %s 2>&1 > dp.out" %parent_model
                slurm_machine.machine_dict['command'] = command
            else:
                pass
            slurm_machine.write(model_dir/'dptrain.slurm')

        return

    def freeze_ensemble(self):
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
                self.__freeze(model_dir)
            self.__freeze(model_dir)

        return

    def check_finished(self, model_path):
        """check if the training is finished"""
        converged = False
        model_path = Path(model_path)
        dpout_path = model_path / 'dp.out'
        if dpout_path.exists():
            content = dpout_path.read_text()
            line = content.split('\n')[-3]
            print(line)
            #if 'finished' in line:
            #    converged = True

        return converged

    def __freeze(self, model_path):
        command = "dp freeze -o graph.pb"
        output = run_command(model_path, command)
        print(output)
        return 


if __name__ == "__main__":
    pass