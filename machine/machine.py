#!/usr/bin/env python3
# -*- coding: utf-8 -*

import json

from abc import ABC
from abc import abstractmethod

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

    registered_keywords = {}

    def __init__(self, machine_json):
        self.prefix = '#SBATCH'

        with open(machine_json, 'r') as fopen:
            self.machine_dict = json.load(fopen)
        
        return

    def __repr__(self):
        machine_dict = self.machine_dict
        content = "#!/bin/bash -l \n"
        content += "#SBATCH --job-name=%s         \n" %(machine_dict['job-name'])
        content += "#SBATCH --partition=%s        \n" %machine_dict['partition']
        content += "#SBATCH --time=%s             \n" %machine_dict['time']
        content += "#SBATCH --nodes=1-%s          \n" %(int(machine_dict['nodes']))
        if machine_dict['mode'] == 'cpu':
            content += "#SBATCH --ntasks=%s           \n" %(int(self.machine_dict['nodes'])*int(self.machine_dict['cpus_per_node']))
            content += "#SBATCH --cpus-per-task=1     \n"
            content += "#SBATCH --mem=10G             \n"
            content += "#SBATCH --output=slurm.o%j    \n"
            content += "#SBATCH --error=slurm.e%j     \n"
        elif machine_dict['mode'] == 'gpu':
            content += "#SBATCH --ntasks=%s           \n" %(machine_dict['ntasks'])
            content += "#SBATCH --cpus-per-task=%s    \n" %(machine_dict['cpus-per-task'])
            content += "#SBATCH --gres=%s             \n" %(machine_dict['gres'])
            content += "#SBATCH --mem-per-cpu=%s      \n" %(machine_dict['mem-per-cpu'])
            content += "#SBATCH --mem-per-gpu=%s      \n" %(machine_dict['mem-per-gpu'])
            content += "#SBATCH --output=slurm.o%j    \n"
            content += "#SBATCH --error=slurm.e%j     \n"
        else:
            raise ValueError('machine mode must be cpu or gpu.')
        content += "\n"
        content += "%s\n" %machine_dict['environs']
        content += "%s\n" %machine_dict['modules']
        content += "\n"
        content += "%s\n" %machine_dict['command']

        return content

    def parse_params(self):
        pass

    def write(self, script_path):
        """"""
        with open(script_path, 'w') as fopen:
            fopen.write(str(self))

        return

    def submit_script(self):
        pass

    def check_status(self):
        pass

class PbsMachine(AbstractMachine):

    def __init__(self):

        pass

    def miaow(self):
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

        return 

if __name__ == '__main__':
    vasp_slurm = SlurmMachine('./machine.json')
    # print(vasp_slurm)
    vasp_slurm.write_script()
    pass