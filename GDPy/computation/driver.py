#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import pathlib

from typing import Optional, NoReturn
from collections.abc import Iterable

import numpy as np

from ase.io import read, write


class AbstractDriver(abc.ABC):

    delete = []
    keyword: Optional[str] = None
    special_keywords = {}

    _directory = pathlib.Path.cwd()

    def __init__(self, calc, params, directory, *args, **kwargs):

        self.calc = calc
        self.calc.reset()

        self.directory = pathlib.Path(directory)

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
    def _parse_params(self, params: dict) -> NoReturn:
        """ parse different tasks, and prepare init and run params
            for each task, different behaviours should be realised in specific object
        """
        task_ = params.pop("task", self.default_task)
        if task_ not in self.supported_tasks:
            raise NotImplementedError(f"{task_} is invalid for {self.__name__}...")

        init_params_ = params.pop("init", {})
        init_params_.update(self.default_init_params[task_])

        run_params_ = params.pop("run", {})
        run_params_.update(self.default_run_params[task_])

        self.task = task_
        self.init_params = init_params_
        self.run_params = run_params_

        return 
    
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
        """"""


        return 


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


def run_driver(potter, params, structure):
    """
    """
    import shutil
    # - parse inputs
    print("potter: ", potter.name)
    from ase.io import read, write
    from GDPy.utils.command import parse_input_file
    params = parse_input_file(params)
    driver = potter.create_driver(params)

    frames = read(structure, ":")
    print("nframes: ", len(frames))

    res_dpath = pathlib.Path.cwd() / "results"
    if res_dpath.exists():
        shutil.rmtree(res_dpath)
        print("remove previous results...")
    res_dpath.mkdir()

    # - run dynamics
    for i, atoms in enumerate(frames):
        driver.directory = res_dpath / ("cand"+str(i))
        driver.run(atoms)

    return


if __name__ == "__main__":
    pass