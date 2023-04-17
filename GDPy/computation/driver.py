#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import dataclasses
import pathlib

from typing import Optional, NoReturn, List
from collections.abc import Iterable

import numpy as np

from ase import Atoms

#: Parameter keys used to init a minimisation task.
MIN_INIT_KEYS: List[str] = ["min_style", "min_modify", "dump_period"]

#: Parameter keys used to run a minimisation task.
MIN_RUN_KEYS: List[str] = ["steps", "fmax"]

#: Parameter keys used to init a molecular-dynamics task.
MD_INIT_KEYS: List[str] = [
    "md_style", "velocity_seed", "timestep", "temp", "Tdamp", 
    "press", "Pdamp", "dump_period"
]

#: Parameter keys used to run a molecular-dynamics task.
MD_RUN_KEYS: List[str] = ["steps"]

@dataclasses.dataclass
class DriverSetting:

    """These are geometric parameters. Electronic?
    """

    #: Simulation task.
    task: str = "min"

    #: 
    min_style: str = "bfgs"
    min_modify: str = "integrator verlet tmax 4",
    maxstep: float = 0.1

    #:
    md_style: str = "nvt"
    velocity_seed: int = None
    timestep: float = 1.0

    temp: float = 300.
    tend: float = None
    Tdamp: float = 100. # fs

    press: float = 1.0 # bar
    pend: float = 1.0 # bar
    Pdamp: float = 100.

    #: Shared parameters among tasks.
    dump_period: int = 1

    #: run params
    etol: float = None # 5e-2
    fmax: float = None # 1e-5
    steps: int = 0

    constraint: str = None

    #: Parameters that are used to update 
    _internals: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """A dummy function that will be overridden by subclasses."""

        return
    
    def get_init_params(self):
        """"""

        return copy.deepcopy(self._internals)
    
    def get_run_params(self, *args, **kwargs):
        """"""
        raise NotImplementedError(f"{self.__class__.__name__} has no function for run params.")

class AbstractDriver(abc.ABC):

    #: Driver's name.
    name: str = "abstract"

    #: Deleted keywords.
    delete: list = []

    #: Keyword.
    keyword: Optional[str] = None

    #: Special keywords.
    special_keywords: dict = {}

    #: Systemwise parameter keys.
    syswise_keys: list = []

    #: Default init parameters.
    default_init_params: dict = dict()

    #: Default run parameters.
    default_run_params: dict = dict()

    #: Map input key names to the actual names used in the backend.
    param_mapping: dict = dict()

    #: Parameters for PotentialManager.
    pot_params: dict = None

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        """Init a driver.

        Args:
            calc: The ase calculator.
            params: Driver parameters.
            directory: Working directory.

        """
        self.calc = calc
        self.calc.reset()

        self._directory = pathlib.Path(directory)

        self._org_params = copy.deepcopy(params)
        self._parse_params(params)

        return
    
    @property
    @abc.abstractmethod
    def default_task(self) -> str:
        """Default simulation task."""

        return

    @property
    @abc.abstractmethod
    def supported_tasks(self) -> List[str]:
        """Supported simulation tasks"""

        return
    
    @property
    def directory(self):
        """Set working directory of this driver.

        Note:
            The attached calculator's directory would be set as well.
        
        """

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        self._directory = pathlib.Path(directory_)
        self.calc.directory = str(self.directory) # NOTE: avoid inconsistent in ASE

        return
    
    @abc.abstractmethod
    def _parse_params(self, params_: dict) -> NoReturn:
        """Parse different tasks, and prepare init and run params.

        For each task, different behaviours should be realised in specific object.

        """
        params = copy.deepcopy(params_)

        task_ = params.pop("task", self.default_task)
        if task_ not in self.supported_tasks:
            raise NotImplementedError(f"{task_} is invalid for {self.__class__.__name__}...")

        # - init
        init_params_ = copy.deepcopy(self.default_init_params[task_])
        kwargs_ = params.pop("init", {})
        init_params_.update(**kwargs_)
        init_params_ = self._map_params(init_params_)

        # - run
        run_params_ = copy.deepcopy(self.default_run_params[task_])
        kwargs_ = params.pop("run", {})
        run_params_.update(**kwargs_)
        run_params_ = self._map_params(run_params_)

        self.task = task_
        self.init_params = init_params_
        self.run_params = run_params_

        return 
    
    def _map_params(self, params):
        """Map params, avoid conflicts."""
        if hasattr(self, "param_mapping"):
            params_ = {}
            for key, value in params.items():
                new_key = self.param_mapping.get(key, None)
                if new_key is not None:
                    key = new_key
                params_[key] = value
        else:
            params_ = params

        return params_
    
    def get(self, key):
        """Get param value from init/run params by a mapped key name."""
        parameters = copy.deepcopy(self.init_params)
        parameters.update(copy.deepcopy(self.run_params))

        value = parameters.get(key, None)
        if not value:
            mapped_key = self.param_mapping.get(key, None)
            if mapped_key:
                value = parameters.get(mapped_key, None)

        return value
    
    def reset(self) -> NoReturn:
        """Remove results stored in dynamics calculator."""
        self.calc.reset()

        return

    def delete_keywords(self, kwargs) -> NoReturn:
        """Removes list of keywords (delete) from kwargs."""
        for d in self.delete:
            kwargs.pop(d, None)

        return

    def set_keywords(self, kwargs) -> NoReturn:
        """Set list of keywords from kwargs."""
        args = kwargs.pop(self.keyword, [])
        if isinstance(args, str):
            args = [args]
        elif isinstance(args, Iterable):
            args = list(args)

        for key, template in self.special_keywords.items():
            if key in kwargs:
                val = kwargs.pop(key)
                args.append(template.format(val))

        kwargs[self.keyword] = args

        return

    @abc.abstractmethod
    def run(self, atoms, read_exists: bool=True, extra_info: dict=None, *args, **kwargs) -> Atoms:
        """Return the last frame of the simulation.

        Copy input atoms, and return a new atoms. Check whether the simulation is
        finished and retrieve stored results. If necessary, extra information could 
        be added to the atoms.info

        """
        new_atoms = copy.deepcopy(atoms)

        return new_atoms

    @abc.abstractmethod
    def read_trajectory(self, *args, **kwargs) -> List[Atoms]:
        """Read trajectory in the current working directory.
        """

        return
    
    def read_converged(self, *args, **kwargs) -> Atoms:
        """Read last frame of the trajectory.

        It would be better if the structure were checked to be converged.

        """
        traj_frames = self.read_trajectory(*args, **kwargs)

        return traj_frames[-1]
    
    def as_dict(self) -> dict:
        """Return parameters of this driver."""
        params = dict(
            backend = self.name
        )
        # NOTE: we use original params otherwise internal param names would be 
        #       written out and make things confusing
        org_params = copy.deepcopy(self._org_params)

        # - update some special parameters
        constraint = self.run_params.get("constraint", None)
        if constraint is not None:
            org_params["run"]["constraint"] = constraint

        params.update(org_params)

        return params


if __name__ == "__main__":
    pass