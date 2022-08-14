#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" worker = drivers + machine
"""

import pathlib
import yaml

import numpy as np

from ase.io import read, write

from GDPy.utils.command import parse_input_file, CustomTimer

from GDPy.machine.factory import create_machine

from GDPy.potential.manager import PotManager

class SpecificWorker():

    """ - components
        single-frame methods
            Monte Carlo
        population methods
            GA = (generator + propagator) + driver + machine
        - machines
            moniter
            job
    """

    mode = "serial" # serial, queue
    batchsize = 1 # how many structures performed in one job

    _directory = pathlib.Path.cwd()
    _machine = None

    prefix = "worker"

    def __init__(self, params) -> None:
        """
        """
        self.params = params

        self.prefix = params.pop("prefix", "worker")
        self.batchsize = params.pop("batchsize", 1)

        # - potter and driver
        pot_dict = params.get("potential", None)
        if pot_dict is None:
            raise RuntimeError("Need potential...")
        pm = PotManager() # main potential manager
        potter = pm.create_potential(pot_name = pot_dict["name"])
        potter.register_calculator(pot_dict["params"])
        potter.version = pot_dict["version"] # NOTE: important for calculation in exp

        self.driver = potter.create_driver(params["driver"])

        # - create machine
        machine_params = params.pop("machine", None)
        self.machine = create_machine(machine_params)

        # - set default directory
        self.directory = self.directory / "MyWorker" # TODO: set dir

        return

    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        self._directory = pathlib.Path(directory_)

        return
    
    @property
    def machine(self):

        return self._machine
    
    @machine.setter
    def machine(self, machine_):
        self._machine = machine_

        return
    
    def run(self, frames=None) -> None:
        """"""
        # - split frames
        nframes = len(frames)
        ngroups = int(np.floor(1.*nframes/self.batchsize))
        group_indices = [0]
        for i in range(ngroups):
            group_indices.append((i+1)*self.batchsize)
        if group_indices[-1] != nframes:
            group_indices.append(nframes)
        #print(group_indices)
        starts, ends = group_indices[:-1], group_indices[1:]
        assert len(starts) == len(ends), "Inconsistent start and end indices..."
        #group_indices = [f"{s}:{e}" for s, e in zip(starts,ends)]
        #print(group_indices)

        # - create main dir
        if not self.directory.exists():
            self.directory.mkdir()
        else:
            pass
        
        finished_wdirs = []

        machine = self.machine
        for i, (s,e) in enumerate(zip(starts,ends)):
            cur_frames = frames[s:e]
            # - prepare machine
            group_directory = self.directory / f"g{i}"
            # - set specific params
            job_name = self.prefix + f"-g{i}"
            machine.set(**{"job-name": job_name})
            machine.script = group_directory/"run-driver.slurm" 

            # - create or check the working directory
            if not group_directory.exists():
                group_directory.mkdir()

                with open(group_directory/"driver.yaml", "w") as fopen:
                    yaml.dump(self.params, fopen)
        
                write(group_directory/"frames.xyz", cur_frames)

                if self.mode == "serial":
                    machine.user_commands = "gdp driver {} -s {}".format(
                        group_directory/"driver.yaml", 
                        group_directory/"frames.xyz"
                    )
                    machine.write()
                    print(f"{group_directory.name}: ", machine.submit())
                #print(self.machine)
            else:
                # - check status and get results
                if machine.is_finished():
                    print(f"{job_name} is finished...")
                    finished_wdirs.append(group_directory)
                else:
                    print(f"{job_name} is running...")
        
        # - read results
        if finished_wdirs:
            self.read_results(finished_wdirs)
        
        return
    
    def read_results(self, wdirs=None):
        """"""
        print("ndirs: ", len(wdirs))

        results = []
        
        driver = self.driver
        with CustomTimer(name="read-results"):
            for wd in wdirs:
                cur_frames = read(wd/"frames.xyz", ":")
                for i, atoms in enumerate(cur_frames):
                    driver.directory = wd / ("cand"+str(i))
                    new_atoms = driver.run(atoms, read_exsits=True)
                    print(new_atoms.get_potential_energy())
                    results.append(new_atoms)
        print(results)

        return


def run_worker(params: str, structure: str, potter=None):
    """"""
    # - check if params are all valid
    params = parse_input_file(params)
    """ # TODO: check params
    if potter is None:
        pot_dict = params.get("potential", None)
        if pot_dict is None:
            raise RuntimeError("Need potential...")
        pm = PotManager() # main potential manager
        potter = pm.create_potential(pot_name = pot_dict["name"])
        potter.register_calculator(pot_dict["params"])
        potter.version = pot_dict["version"] # NOTE: important for calculation in exp
    """

    worker = SpecificWorker(params)

    # TODO: 
    frames = read(structure, ":")

    # - find input frames
    worker.run(frames)

    return


if __name__ == "__main__":
    pass