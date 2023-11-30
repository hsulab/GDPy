#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import logging
import os
import shutil
import pathlib
from typing import NoReturn, List

import numpy as np

from ase import Atoms
from ase.io import read, write

from gdpx.core.register import registers
from gdpx.core.variable import Variable
from gdpx.worker.single import SingleWorker
from ..expedition import AbstractExpedition
from .operators import select_operator, parse_operators, save_operator, load_operator

"""This module tries to offer a base class for all MonteCarlo-like methods.
"""

class MonteCarloVariable(Variable):

    def __init__(self, builder, worker, directory="./", *args, **kwargs) -> None:
        """"""
        # - builder
        if isinstance(builder, dict):
            builder_params = copy.deepcopy(builder)
            builder_method = builder_params.pop("method")
            builder = registers.create(
                "builder", builder_method, convert_name=False, **builder_params
            )
        else: # variable
            builder = builder.value
        # - worker
        if isinstance(worker, dict):
            worker_params = copy.deepcopy(worker)
            worker = registers.create("variable", "computer", convert_name=True, **worker_params).value[0]
        elif isinstance(worker, Variable): # computer variable
            worker = worker.value[0]
        elif isinstance(worker, SingleWorker): # assume it is a DriverBasedWorker
            worker = worker
        else:
            raise RuntimeError(f"MonteCarlo needs a SingleWorker instead of a {worker}")
        engine = self._create_engine(builder, worker, *args, **kwargs)
        engine.directory = directory
        super().__init__(initial_value=engine, directory=directory)

        return
    
    def _create_engine(self, builder, worker, *args, **kwargs) -> None:
        """"""
        engine = MonteCarlo(builder, worker, *args, **kwargs)

        return engine


class MonteCarlo(AbstractExpedition):

    restart = False

    #: Prefix of the working directory.
    WDIR_PREFIX: str = "cand"

    #: Name of the MC trajectory.
    TRAJ_NAME: str = "mc.xyz"

    #: Name of the file stores MC information (operations).
    INFO_NAME: str = "opstat.txt"

    def __init__(
        self, builder: dict, worker: dict, operators: List[dict], convergence: dict,
        random_seed=None, dump_period: int=1, restart: bool=False, 
        directory="./", *args, **kwargs
    ) -> None:
        """Parameters for Monte Carlo.

        Args:
            overwrite: Whether overwrite calculation directory.
        
        """
        self.directory = directory
        self.dump_period = dump_period
        self.restart = restart

        # - set random seed that 
        if random_seed is None:
            random_seed = np.random.randint(0, 10000)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)

        # - check system type
        if isinstance(builder, dict):
            builder_params = copy.deepcopy(builder)
            builder_method = builder_params.pop("method")
            builder = registers.create(
                "builder", builder_method, convert_name=False, **builder_params
            )
        else:
            builder = builder
        self.builder = builder

        frames = builder.run()
        assert len(frames) == 1, f"{self.__class__.__name__} only accepts one structure."
        self.atoms = frames[0]

        # - create worker
        if isinstance(worker, dict):
            worker_params = copy.deepcopy(worker)
            worker = registers.create("variable", "computer", convert_name=True, **worker_params).value[0]
        else:
            ...
        assert isinstance(worker, SingleWorker), f"{self.__class__.__name__} only supports SingleWorker (set use_single=True)."
        self.worker = worker

        # - parse operators
        self.operators, self.op_probs = parse_operators(operators)

        # - parse convergence
        self.convergence = convergence
        if self.convergence.get("steps", None) is None:
            self.convergence["steps"] = 1

        return

    def _init_structure(self):
        """Initialise the input structure.

        Set proper tags and minimise the structure. Prepare `self.atoms`, 
        `self.energy_stored`, and `self.curr_step`.
        
        """
        step_wdir = self.directory / f"{self.WDIR_PREFIX}0"
        if not step_wdir.exists():
            # - prepare atoms
            self._print("===== MonteCarlo Structure =====\n")
            tags = self.atoms.arrays.get("tags", None)
            if tags is None:
                # default is setting tags by elements
                symbols = self.atoms.get_chemical_symbols()
                type_list = sorted(list(set(symbols)))
                new_tags = [type_list.index(s)*10000+i for i, s in enumerate(symbols)]
                self.atoms.set_tags(new_tags)
                self._print("set default tags by chemical symbols...")
            else:
                self._print("set attached tags from the structure...")

        # - run init
        self._print("===== MonteCarlo Initial Minimisation =====\n")
        # NOTE: atoms lost tags in optimisation
        #       TODO: move this part to driver?
        curr_tags = self.atoms.get_tags()

        self.atoms.info["confid"] = 0
        self.atoms.info["step"] = -1 # NOTE: remove step info

        # TODO: whether init driver?
        self.worker.directory = step_wdir
        _ = self.worker.run([self.atoms])
        self.worker.inspect(resubmit=True)
        if self.worker.get_number_of_running_jobs() == 0:
            curr_frames = self.worker.retrieve()[0]
            # - update atoms
            curr_atoms = curr_frames[-1]
            self.energy_stored = curr_atoms.get_potential_energy()
            self._print(f"ene: {self.energy_stored}")
            self.atoms = curr_atoms
            self.atoms.set_tags(curr_tags)
            write(self.directory/self.TRAJ_NAME, self.atoms)

            # - 
            self.curr_step = 0

            # - log operator status
            with open(self.directory/self.INFO_NAME, "w") as fopen:
                fopen.write(
                    "{:<24s}  {:<12s}  {:<12s}  {:<12s}  {:<12s}  \n".format(
                        "#Operator", "natoms", "Success", "prev_ene", "curr_ene"
                    )
                )
            step_converged = True
        else:
            step_converged = False

        return step_converged
    
    def _restart_structure(self):
        """Restart from the saved structure.

        Prepare `self.atoms`, `self.energy_stored`, and `self.curr_step`.
        
        """
        mctraj = read(self.directory/self.TRAJ_NAME, ":")
        nframes = len(mctraj)
        self.curr_step = nframes
        self.atoms = mctraj[-1]
        self.energy_stored = self.atoms.get_potential_energy()

        step_converged = True

        return step_converged

    def run(self, *args, **kwargs):
        """Run MonteCarlo simulation."""
        # - some imported packages change `logging.basicConfig` 
        #   and accidently add a StreamHandler to logging.root
        #   so remove it...
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logging.root.removeHandler(h)

        # - prepare logger and output some basic info...
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        self._print("===== MonteCarlo Operators (Modifiers) =====\n")
        for op in self.operators:
            self._print(str(op))
        self._print(f"normalised probabilities {self.op_probs}\n")

        # -- add print function to operators
        for op in self.operators:
            op._print = self._print

        # NOTE: check if operators' regions are consistent
        #       though it works, unexpected results may occur
        # TODO: need rewrite eq function as compare array is difficult
        #noperators = len(self.operators)
        #for i in range(1,noperators):
        #    if self.operators[i].region != self.operators[i-1].region:
        #        raise RuntimeError(f"Inconsistent region found in op {i-1} and op {i}")

        # - init structure
        step_converged = False
        if not (self.directory/self.TRAJ_NAME).exists():
            step_converged = self._init_structure()
        else:
            step_converged = self._restart_structure()
        
        if not step_converged:
            self._print("Wait structure to initialise.")
            return
        else:
            self.curr_step += 1
        
        # - run mc steps
        step_converged = False
        for i in range(self.curr_step, self.convergence["steps"]+1):
            step_converged = self._irun(i)
            if not step_converged:
                self._print("Wait MC step to finish.")
                break
        else:
            self._print("MC is converged...")

        return
    
    def _irun(self, i):
        """Run a single MC step."""
        self._print(f"===== MC Step {i} =====")
        step_wdir = self.directory/f"{self.WDIR_PREFIX}{i}"
        self.worker.directory = step_wdir

        # - operate atoms
        #    always return atoms even if no change is applied
        temp_op = self.directory/f"op_{i}.pkl"
        temp_stru = self.directory/f"temp_{i}.xyz"
        if not temp_stru.exists():
            curr_op = select_operator(self.operators, self.op_probs, self.rng)
            self._print(f"operator {curr_op.__class__.__name__}")
            curr_atoms = curr_op.run(self.atoms, self.rng)
            if curr_atoms:
                save_operator(curr_op, temp_op)
                # --- add info
                curr_atoms.info["confid"] = int(f"{i}")
                curr_atoms.info["step"] = -1 # NOTE: remove step info from driver
                write(temp_stru, curr_atoms)
            else:
                #success = False # fail to run postprocess
                raise RuntimeError("failed to run operation...")
        else:
            # load state from file
            curr_op = load_operator(temp_op)
            curr_atoms = read(temp_stru)
        
        # - TODO: save some info not stored by driver
        curr_tags = curr_atoms.get_tags()

        # - run postprocess (spc, min or md)
        _ = self.worker.run([curr_atoms], read_exists=True)
        self.worker.inspect(resubmit=True)
        if self.worker.get_number_of_running_jobs() == 0:
            curr_atoms = self.worker.retrieve()[0][-1]
            curr_atoms.set_tags(curr_tags)

            self.energy_operated = curr_atoms.get_potential_energy()
            self._print(f"post ene: {self.energy_operated}")

            # -- metropolis
            success = curr_op.metropolis(
                self.energy_stored, self.energy_operated, self.rng
            )

            with open(self.directory/self.INFO_NAME, "a") as fopen:
                fopen.write(
                    "{:<24s}  {:<12d}  {:<12s}  {:<12.4f}  {:<12.4f}  \n".format(
                        curr_op.__class__.__name__, len(self.atoms), str(success), 
                        self.energy_stored, self.energy_operated
                    )
                )

            # -- update atoms
            if success:
                self.energy_stored = self.energy_operated
                self.atoms = curr_atoms
                self._print("success...")
            else:
                self._print("failure...")
            write(self.directory/self.TRAJ_NAME, self.atoms, append=True)

            # -- clean up
            #import psutil
            #proc = psutil.Process()
            #for p in proc.open_files():
            #    print(p)
            os.remove(temp_stru)
            os.remove(temp_op)
            if (i%self.dump_period != 0):
                shutil.rmtree(self.directory/f"{self.WDIR_PREFIX}{i}")
            
            step_converged = True
        else:
            step_converged = False

        return step_converged
    
    def read_convergence(self):
        """Check the convergence of MC.

        Currently, the only criteria is whether the simulation reaches the maximum 
        steps.

        """
        converged = False
        if (self.directory/self.TRAJ_NAME).exists():
            mctraj = read(self.directory/self.TRAJ_NAME, ":")
            nframes = len(mctraj)
            #self.curr_step = nframes
            if nframes > self.convergence["steps"]:
                converged = True
        else:
            ...

        return converged
    
    def get_workers(self):
        """Get all workers used by this expedition."""
        wdirs = list(self.directory.glob(f"{self.WDIR_PREFIX}*"))
        wdirs = sorted(wdirs, key=lambda x: int(x.name[len(self.WDIR_PREFIX):]))

        workers = []
        for curr_wdir in wdirs:
            curr_worker = copy.deepcopy(self.worker)
            curr_worker.directory = curr_wdir
            workers.append(curr_worker)

        return workers

    def as_dict(self) -> dict:
        """"""
        engine_params = {}
        engine_params["method"] = "monte_carlo"
        engine_params["builder"] = self.builder.as_dict()
        engine_params["worker"] = self.worker.as_dict()
        engine_params["operators"] = []
        for op in self.operators:
            engine_params["operators"].append(op.as_dict())
        engine_params["dump_period"] = self.dump_period
        engine_params["convergence"] = self.convergence
        engine_params["random_seed"] = self.random_seed

        engine_params = copy.deepcopy(engine_params)

        return engine_params



if __name__ == "__main__":
    ...
