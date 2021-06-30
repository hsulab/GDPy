#!/usr/bin/env python3
# -*- coding: utf-8 -*

from os import register_at_fork
import re
import json
import pathlib

from collections import OrderedDict

from abc import ABC
from abc import abstractmethod
from typing import NoReturn

class AbstractMachine(ABC):

    def write_script(self):
        pass

    @abstractmethod
    def parse_params(self, *args, **kwargs):
        pass

    def register_mode(self, *args):
        pass

class LocalMachine(AbstractMachine):

    def __init__(self):
        pass

    def parse_params(self):
        pass

class SlurmMachine(AbstractMachine):

    """
    machine parameters
    module and environment
    executables
    """

    PREFIX = "#SBATCH"
    SUFFIX = ".slurm"
    SHELL = "#!/bin/bash -l"

    registered_keywords = [
        'job-name',
        'partition',
        'time',
        'nodes',
        'ntasks',
        'cpus-per-task',
        'gres',
        'output',
        'error'
    ]

    default_params = {
        'job-name': 'slurm',
        'partition': None,
        'time': None,
        'nodes': None,
        'ntasks': None,
        'cpus-per-task': None,
        'output': 'slurm.o%j',
        'error': 'slurm.e%j'
    }

    extra_gpu_params = {
        'gres': None
    }

    user_commands = None

    def __init__(self, machine_json):
        """"""
        self.machine_dict = self.default_params.copy()

        machine_json = pathlib.Path(machine_json)

        if machine_json.suffix == 'json':
            with open(machine_json, 'r') as fopen:
                self.machine_dict = json.load(fopen)
        elif machine_json.suffix == self.SUFFIX:
            self.read(machine_json)
        
        return

    def __str__(self):
        content = self.SHELL + '\n'
        for key, value in self.machine_dict.items():
            if value:
                content += '{} --{}={}\n'.format(self.PREFIX, key, value)
            else:
                raise ValueError('Keyword %s not properly set.' %key)
        
        if self.user_commands:
            content += self.user_commands
        else:
            raise ValueError('not initialise properly')

        return content

    def parse_params(self):
        pass

    def read(self, slurm_file):
        """ read slurm file and update machine params
        """
        with open(slurm_file, 'r') as fopen:
            lines = fopen.readlines()
        
        # find keywords
        sbatch_lines, command_lines = [], []
        for line in lines[1:]: # the first line should #!
            if line.strip():
                if re.match(self.PREFIX, line):
                    sbatch_lines.append(line)
                else:
                    command_lines.append(line)
        
        # update params with sbatch lines
        for line in sbatch_lines:
            sbatch_data = line.strip().split()[1] # pattern: #SBATCH --option=value 
            key, value = sbatch_data.strip().split('=')
            if key.startswith('--'):
                key = key.strip('--')
                self.machine_dict[key] = value
            elif key.startswith('-'):
                raise NotImplementedError('Unknown keyword %s' %key)
            else:
                raise ValueError('Unknown keyword %s' %key)
        
        # update user commands
        self.user_commands = ''.join(command_lines)

        return

    def write(self, script_path):
        """"""
        with open(script_path, 'w') as fopen:
            fopen.write(str(self))

        return

    def submit_script(self):
        """ submit jobs and taks job ids
        """
        pass

    def check_status(self):
        pass

class PbsMachine(AbstractMachine):

    PREFIX = "#$"
    SUFFIX = ".slurm"
    SHELL = "#!/bin/bash -l"

    def __init__(self):

        return

    def parse_params(self):

        return

    def __str__(self):
        content = "#!/bin/bash -l\n"
        content += "#$ -N %s\n" %Path(directory).name
        content += "#$ -l h_rt=24:00:00\n"
        content += "#$ -l mem=1G\n"
        content += "#$ -pe mpi %s\n"  %ncpus
        content += "#$ -cwd \n"
        content += "#$ -P Gold\n"
        content += "#$ -A QUB_chem\n"
        content += "\n"
        content += "gerun /home/mmm0586/apps/vasp/installed/5.4.1-TS/vasp_std 2>&1 > vasp.out\n"

        return content

if __name__ == '__main__':
    vasp_slurm = SlurmMachine('../../templates/inputs/machine.json')
    vasp_slurm.read('../../templates/jobscripts/vasp.slurm')
    print(vasp_slurm)
    #vasp_slurm.write_script()
    pass