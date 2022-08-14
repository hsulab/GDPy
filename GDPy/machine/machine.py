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

    default_parameters = {}
    parameters = {}

    _script = None

    environs = None
    user_commands = None

    status = []

    @property
    def script(self):

        return self._script
    
    @script.setter
    def script(self, script_):
        self._script = pathlib.Path(script_)
        return 

    def _get_default_parameters(self):
        return self.default_parameters.copy()

    def set(self, **kwargs) -> dict:
        """"""
        #changed_parameters = {}
        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            #if key not in self.parameters or not equal(value, oldvalue):
            #    changed_parameters[key] = value
            #    self.parameters[key] = value
            self.parameters[key] = value

        return

    def write_script(self):
        pass

    def register_mode(self, *args):
        return

    def write(self):
        """"""
        with open(self.script, "w") as fopen:
            fopen.write(str(self))

        return

    def submit(self):
        """submit job using specific machine command"""
        # submit job
        command = "{0} {1}".format(self.SUBMIT_COMMAND, self.script.name)
        proc = subprocess.Popen(
            command, shell=True, cwd=self.script.parent,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding = "utf-8"
        )
        errorcode = proc.wait(timeout=10) # 10 seconds
        if errorcode:
            raise RuntimeError(f"Error in submitting job script {str(self.script)}")

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
    ENQUIRE_COMMAND = "`which squeue` -u `whoami` --format=\"%.12i %.12P %.24j %.4t %.12M %.12L %.5D %.4C\""

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
    default_parameters = {
        "job-name": "slurmJob",
        "account": None,
        "partition": None,
        "time": None,
        # - CPU
        "nodes": None,
        "ntasks": None,
        "tasks-per-node": None,
        "cpus-per-task": None,
        "mem-per-cpu": None, #"4G"
        #"output": "slurm.o%j",
        #"error": "slurm.e%j"
        # - GPU
        "gres": None,
        "mem-per-gpu": None # "32G"
    }

    status = ["R", "Q", "PD", "CG"]
    running_status = ["R", "Q", "PD"]

    def __init__(self, *args, **kwargs):
        """"""
        # - update params
        self.environs = kwargs.pop("environs", "")
        self.user_commands = kwargs.pop("user_commands", "")

        # - make default params
        self.parameters = self._get_default_parameters()
        #parameters_ = kwargs.pop("parameters", None)
        #if parameters_:
        #    self.parameters.update(parameters_)
        self.parameters.update(kwargs)
        
        return

    def __str__(self):
        """job script"""
        # - slurm params
        content = self.SHELL + "\n"
        for key, value in self.parameters.items():
            if value:
                content += "{} --{}={}\n".format(self.PREFIX, key, value)
            #else:
            #    raise ValueError("Keyword *%s* not properly set." %key)
        
        if self.environs:
            content += "\n\n"
            content += self.environs
        
        if self.user_commands:
            content += "\n\n"
            content += self.user_commands

        return content

    def update(self, input_params):
        """update machine parameters"""
        machine_dict = parse_input_file(input_params) # dict, json, yaml

        if isinstance(machine_dict, dict):
            # TODO: set kwargs instead of overwrite
            # self.machine_dict = machine_dict
            self.machine_dict.update(**machine_dict)
        elif input_params.suffix == self.SUFFIX:
            self._read(input_params)

        return

    def _read(self, slurm_file):
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

    def is_finished(self) -> bool:
        """"""
        # - run enquire
        p = subprocess.Popen(
            [self.ENQUIRE_COMMAND],
            shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            close_fds=True, universal_newlines=True
        )
        fout = p.stdout
        lines = fout.readlines()

        # - run over results
        finished = False
        for line in lines[1:]: # skipe first info line
            data = line.strip().split()
            jobid, name, status = data[0], data[2], data[3]
            #if name.startswith(self.prefix) and status in self.running_status:
            #    indices = re.match(self.prefix+"*", name).span()
            #    if indices is not None:
            #        confid = int(name[indices[1]:])
            #    confids.append(int(confid))
            if name == self.parameters["job-name"]:
                finished = False
                break
        else:
            finished = True

        return finished


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