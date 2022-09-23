#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import pathlib

from typing import Optional, NoReturn, List
from collections.abc import Iterable

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.utils.command import CustomTimer


class AbstractDriver(abc.ABC):

    delete = []
    keyword: Optional[str] = None
    special_keywords = {}

    syswise_keys = []

    pot_params = None

    def __init__(self, calc, params, directory=pathlib.Path.cwd(), *args, **kwargs):

        self.calc = calc
        self.calc.reset()

        self._directory = pathlib.Path(directory)

        self._parse_params(params)

        return
    
    @property
    @abc.abstractmethod
    def default_task(self):

        return

    @property
    @abc.abstractmethod
    def supported_tasks(self):

        return
    
    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        self._directory = pathlib.Path(directory_)
        self.calc.directory = str(self.directory) # NOTE: avoid inconsistent in ASE

        return
    
    @abc.abstractmethod
    def _parse_params(self, params_: dict) -> NoReturn:
        """ parse different tasks, and prepare init and run params
            for each task, different behaviours should be realised in specific object
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
        """ map params, avoid conflicts
        """
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
        """ get param value from init/run params
            by a mapped key name
        """
        parameters = copy.deepcopy(self.init_params)
        parameters.update(copy.deepcopy(self.run_params))

        value = parameters.get(key, None)
        if not value:
            mapped_key = self.param_mapping.get(key, None)
            if mapped_key:
                value = parameters.get(mapped_key, None)

        return value
    
    def reset(self):
        """ remove results stored in dynamics calculator
        """
        self.calc.reset()

        return

    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)

        return

    def set_keywords(self, kwargs):
        # TODO: rewrite this method
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
    def run(self, atoms, **kwargs):
        """ whether return atoms or the entire trajectory
            copy input atoms, and return a new atoms
        """

        return 

    @abc.abstractmethod
    def read_trajectory(self, *args, **kwargs) -> List[Atoms]:
        """ read trajectory in the current working directory
        """

        return
    
    def read_converged(self, *args, **kwargs) -> Atoms:
        """ read last frame of the trajectory
            should better be converged
        """
        traj_frames = self.read_trajectory(*args, **kwargs)

        return traj_frames[-1]
    
    def as_dict(self):
        """"""
        params = dict(
            backend = self.name,
            task = self.task,
            init = copy.deepcopy(self.init_params),
            run = copy.deepcopy(self.run_params)
        )

        return params


def read_trajectories(
    action, tmp_folder, traj_period,
    traj_frames_path, traj_indices_path,
    opt_frames_path
):
    """ read trajectories from several directories
        each dir is named by candx
    """
    # - act, retrieve trajectory frames
    # TODO: more general interface not limited to dynamics
    if not traj_frames_path.exists():
        traj_indices = [] # use traj indices to mark selected traj frames
        all_traj_frames = []
        optimised_frames = read(opt_frames_path, ":")
        # TODO: change this to joblib
        for atoms in optimised_frames:
            # --- read confid and parse corresponding trajectory
            confid = atoms.info["confid"]
            action.set_output_path(tmp_folder/("cand"+str(confid)))
            traj_frames = action._read_trajectory(atoms, label_steps=True)
            # --- generate indices
            # NOTE: last one should be always included since it may be converged structure
            cur_nframes = len(all_traj_frames)
            cur_indices = list(range(0,len(traj_frames)-1,traj_period)) + [len(traj_frames)-1]
            cur_indices = [c+cur_nframes for c in cur_indices]
            # --- add frames
            traj_indices.extend(cur_indices)
            all_traj_frames.extend(traj_frames)
        np.save(traj_indices_path, traj_indices)
        write(traj_frames_path, all_traj_frames)
    else:
        all_traj_frames = read(traj_frames_path, ":")
    print("ntrajframes: ", len(all_traj_frames))
            
    if traj_indices_path.exists():
        traj_indices = np.load(traj_indices_path)
        all_traj_frames = [all_traj_frames[i] for i in traj_indices]
        #print(traj_indices)
    print("ntrajframes: ", len(all_traj_frames), f" by {traj_period} traj_period")

    return all_traj_frames


def run_driver(params, structure, directory=pathlib.Path.cwd(), potter = None):
    """
    """
    import shutil
    from ase.io import read, write
    from GDPy.utils.command import parse_input_file
    from GDPy.potential.register import PotentialRegister

    params = parse_input_file(params)

    # - read structures
    from GDPy.builder import create_generator
    generator = create_generator(structure)
    frames = generator.run()
    nframes = len(frames)
    print("nframes: ", nframes)

    wdirs = params.pop("wdirs", None)
    if wdirs is None:
        wdirs = [f"cand{i}" for i in range(nframes)]

    # - parse inputs
    if potter is None:
        pot_dict = params.pop("potential", None)
        if pot_dict is None:
            raise RuntimeError("Need potential...")
        pm = PotentialRegister() # main potential manager
        potter = pm.create_potential(pot_name = pot_dict["name"])
        potter.register_calculator(pot_dict["params"])
        potter.version = pot_dict.get("version", "unknown") # NOTE: important for calculation in exp
    print("potter: ", potter.name)
    driver = potter.create_driver(params["driver"])

    #res_dpath = pathlib.Path.cwd() / "results"
    #if res_dpath.exists():
    #    shutil.rmtree(res_dpath)
    #    print("remove previous results...")
    #res_dpath.mkdir()

    # - run dynamics
    new_frames = []
    with CustomTimer(name="run-driver"):
        for wdir, atoms in zip(wdirs, frames):
            #print(wdirs, atoms)
            driver.reset()
            driver.directory = directory / wdir
            new_atoms = driver.run(atoms)
            new_frames.append(new_atoms)
    
    # - report
    energies = [a.get_potential_energy() for a in new_frames]
    print(energies)

    return


if __name__ == "__main__":
    pass