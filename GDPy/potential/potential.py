#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
Potential Manager
deals with various machine learning potentials
"""

import abc
import pathlib
from sys import implementation, path
from typing import Union

from GDPy.utils.command import run_command


class AbstractPotential(abc.ABC):
    """
    Create various potential instances
    """

    def __init__(self):
        """
        """

        self.uncertainty = None # uncertainty estimation method

        return
    
    @abc.abstractmethod
    def generate_calculator(self):
        """ generate ase wrapped calculator
        """

        return 

class RXManager(AbstractPotential):

    name = 'RX'
    implemented_backends = ['lammps']

    def __init__(self, backend: str, models: str, type_map: dict):
        """"""
        self.model = models
        self.type_map = type_map

        return

    def generate_calculator(self):
        """ generate calculator with various backends
        """
        if self.backend == 'lammps':
            # return reax pair related content
            content = "units           real\n"
            content += "atom_style      charge\n"
            content += "\n"
            content += "neighbor            1.0 bin\n" # for npt, ghost atom issue
            content += "neigh_modify     every 10 delay 0 check no\n"
            content += 'pair_style  reax/c NULL\n'
            content += 'pair_coeff  * * %s %s\n' %(self.model, ' '.join(list(self.type_map.keys())))
            content += "fix             2 all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
            calc = content
        else:
            pass

        return calc

class DPManager(AbstractPotential):

    name = 'DP'
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
            pot_path = pathlib.Path(self.models)
            pot_dir, pot_pattern = pot_path.parent, pot_path.name
            models = []
            for pot in pot_dir.glob(pot_pattern):
                models.append(str(pot/'graph.pb'))
            self.models = models
        else:
            for m in self.models:
                if not pathlib.Path(m).exists():
                    raise ValueError('Model %s does not exist.' %m)

        return
    
    def _check_uncertainty_support(self):
        """"""
        self.uncertainty = False
        if len(self.models) > 1:
            self.uncertainty = True

        return
    
    def generate_calculator(self):
        """ generate calculator with various backends
        """
        if self.backend == 'ase':
            # return ase calculator
            from GDPy.calculator.dp import DP
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
        model_path = pathlib.Path(model_path)
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

class EANNManager(AbstractPotential):

    name = "EANN"
    implemented_backends = ["ase", "lammps"]

    def __init__(self, backend: str, models: Union[str, list], type_map: dict):
        """ create a eann manager
        """
        self.backend = backend
        if self.backend not in self.implemented_backends:
            raise NotImplementedError('Backend %s is not implemented.' %self.backend)

        # check models
        self.models = models
        self.__parse_models()
        self.__check_uncertainty_support()

        self.type_map = type_map

        return
    
    def __parse_models(self):
        """"""
        if isinstance(self.models, str):
            pot_path = pathlib.Path(self.models)
            pot_dir, pot_pattern = pot_path.parent, pot_path.name
            models = []
            for pot in pot_dir.glob(pot_pattern):
                models.append(str(pot))
            self.models = models
        else:
            for m in self.models:
                if not pathlib.Path(m).exists():
                    raise ValueError('Model %s does not exist.' %m)

        return
    
    def __check_uncertainty_support(self):
        """"""
        self.uncertainty = False
        if len(self.models) > 1:
            self.uncertainty = True

        return
    
    def generate_calculator(self, atypes=None):
        """ generate calculator with various backends
        """
        if self.backend == 'ase':
            # return ase calculator
            #from GDPy.calculator.dp import DP
            #calc = DP(model=self.models, type_dict=self.type_map)
            pass
        elif self.backend == "lammps":
            # return deepmd pair related content
            #content = "units           metal\n"
            #content += "atom_style      atomic\n"
            content = "neighbor        0.0 bin\n"
            content += "pair_style      eann %s \n" \
                %(' '.join([m for m in self.models]))
            content += "pair_coeff * * double %s" %(" ".join(atypes))
            calc = content

        return calc
    
    def create_ensemble(self):

        return

    def freeze_ensemble(self):
        """freeze model"""
        # find models
        cwd = pathlib.Path.cwd()
        model_dirs = []
        for p in cwd.glob("model*"):
            model_dirs.append(p)
        model_dirs.sort()

        print(model_dirs)

        # freeze models
        for model_dir in model_dirs:
            #if self.check_finished(model_dir):
            #    self.freeze_model(model_dir)
            self.__freeze(model_dir)

        return
    
    def __freeze(self, model_path):
        """freeze single model"""
        # find best
        best_path = list(model_path.glob("*BEST*"))
        assert len(best_path) == 1, "there are two or more best models..."
        best_path = best_path[0]
        print(best_path)

        # TODO: change later
        command = "python -u /users/40247882/repository/EANN/eann freeze -pt {0} -o eann_best_".format(
            best_path
        )
        output = run_command(model_path, command)
        print(output)
        return 


if __name__ == '__main__':
    pass