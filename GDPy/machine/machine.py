#!/usr/bin/env python3
# -*- coding: utf-8 -*

from importlib.resources import path
import re
import subprocess
import json
import pathlib

from collections import OrderedDict

from abc import ABC
from abc import abstractmethod
from typing import NoReturn

from GDPy.utils.command import parse_input_file

class AbstractMachine(ABC):

    PREFIX = None
    SUFFIX = None
    SHELL = None
    SUBMIT_COMMAND = None

    def write_script(self):
        pass

    @abstractmethod
    def parse_params(self, *args, **kwargs):
        return

    def register_mode(self, *args):
        return

    def submit(self, script_path):
        """submit job using specific machine command"""
        # submit job
        command = "{0} {1}".format(self.SUBMIT_COMMAND, script_path.name)
        proc = subprocess.Popen(
            command, shell=True, cwd=script_path.parent,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding = "utf-8"
        )
        errorcode = proc.wait(timeout=10) # 10 seconds
        if errorcode:
            raise ValueError("Error in submit job script %s" %script_path)

        output = "".join(proc.stdout.readlines())

        return output

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
    SUBMIT_COMMAND = "sbatch"

    # full name starts with --
    full_keywords = [ 
        "job-name",
        "partition",
        "time",
        "nodes",
        "ntasks",
        "cpus-per-task",
        "mem-per-cpu",
        "gres",
        "output",
        "error"
    ]

    # compability for different machine
    # not all keywords are necessary
    default_cpu_parameters = {
        "job-name": "slurmJob",
        "partition": None,
        "time": None
    }

    __default_cpu_parameters = {
        "job-name": "slurmJob",
        "account": None,
        "partition": None,
        "qos": None,
        "time": None,
        "nodes": None,
        "ntasks": None,
        "tasks-per-node": None,
        "cpus-per-task": None,
        "mem-per-cpu": "4G",
        "output": "slurm.o%j",
        "error": "slurm.e%j"
    }

    extra_gpu_parameters = {
        "gres": None,
        "mem-per-gpu": "32G"
    }

    status = ["R", "Q", "PD", "CG"]

    environs = None
    user_commands = None

    def __init__(self, use_gpu=False, **kwargs):
        """"""
        # make default machine dict
        self.machine_dict = self.default_cpu_parameters.copy()
        if use_gpu:
            self.machine_dict.update(self.extra_gpu_parameters)
        
        # - update params
        self.environs = kwargs.pop("environs", None)
        self.user_commands = kwargs.pop("user_commands", None)
        self.machine_dict.update(**kwargs)
        
        return

    def __str__(self):
        """job script"""
        # - slurm params
        content = self.SHELL + '\n'
        for key, value in self.machine_dict.items():
            if value:
                content += "{} --{}={}\n".format(self.PREFIX, key, value)
            else:
                raise ValueError("Keyword *%s* not properly set." %key)
        
        if self.environs:
            content += "\n\n"
            content += self.environs
        
        if self.user_commands:
            content += "\n\n"
            content += self.user_commands
        else:
            raise ValueError("No user commands.")

        return content

    def parse_params(self):

        return
    
    def update(self, machine_input):
        """update machine parameters"""
        machine_input = pathlib.Path(machine_input)
        machine_dict = parse_input_file(machine_input)

        if machine_dict:
            # TODO: set kwargs instead of overwrite
            # self.machine_dict = machine_dict
            pass
        elif machine_input.suffix == self.SUFFIX:
            self.__read(machine_input)

        return

    def __read(self, slurm_file):
        """ read slurm file and update machine params
            for now, only support keywords start from --
        """
        # read file
        with open(slurm_file, "r") as fopen:
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
            key, value = sbatch_data.strip().split("=")
            if key.startswith("--"):
                key = key.strip("--")
                self.machine_dict[key] = value
            elif key.startswith("-"):
                raise NotImplementedError("Not support abbrev keyword %s" %key)
            else:
                raise ValueError("Wrong format keyword %s" %key)
        
        # TODO: module lines
        # TODO: environs start with export
        # TODO: python lines start with conda
        
        # update user commands
        self.user_commands = "".join(command_lines)

        return

    def write(self, script_path):
        """"""
        with open(script_path, "w") as fopen:
            fopen.write(str(self))

        return

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

if __name__ == "__main__":
    # test slurm machine
    #vasp_slurm = SlurmMachine(use_gpu=False)
    #vasp_slurm.update("/users/40247882/scratch2/alumina-revised/GA/run-vasp/vasp.slurm")
    #print(vasp_slurm)
    # test slurm on archer2
    vasp_slurm = SlurmMachine(use_gpu=False)
    vasp_slurm.update("/mnt/scratch2/users/40247882/alumina-revised/GA/GA-Test/vasp.slurm")
    print(vasp_slurm)