#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Counter, Union, List
import numpy as np

import dataclasses
from dataclasses import dataclass, field

from ase import Atoms
from ase.io import read, write

from GDPy.utils.command import parse_input_file, convert_indices, CustomTimer

from GDPy.expedition.abstract import AbstractExplorer


@dataclasses.dataclass
class MDParams:        

    #unit = "ase"
    task: str = "md"
    md_style: str = "nvt" # nve, nvt, npt

    steps: int = 0 
    dump_period: int = 1 
    timestep: float = 2 # fs
    temp: float = 300 # Kelvin
    pres: float = -1 # bar

    # fix nvt/npt/nph
    Tdamp: float = 100 # fs
    Pdamp: float = 500 # fs

    def __post_init__(self):
        """ unit convertor
        """

        return

def create_dataclass_from_dict(dcls: dataclasses.dataclass, params: dict) -> List[dataclasses.dataclass]:
    """ create a series of dcls instances
    """
    # NOTE: onlt support one param by list
    # - find longest params
    plengths = []
    for k, v in params.items():
        if isinstance(v, list):
            n = len(v)
        else: # int, float, string
            n = 1
        plengths.append((k,n))
    plengths = sorted(plengths, key=lambda x:x[1])
    # NOTE: check only has one list params
    assert sum([p[1] > 1 for p in plengths]) <= 1, "only accept one param as list."

    # - convert to dataclass
    dcls_list = []
    maxname, maxlength = plengths[-1]
    for i in range(maxlength):
        cur_params = {}
        for k, n in plengths:
            if n > 1:
                v = params[k][i]
            else:
                v = params[k]
            cur_params[k] = v
        dcls_list.append(dcls(**cur_params))

    return dcls_list


class MDBasedExpedition(AbstractExplorer):

    """
    Exploration Strategies
        brute-force molecular dynamics
        biased molecular dynamics
    
    Initial Systems
        initial structures must be manually prepared
    
    Units
        fs, eV, eV/AA
    """

    method = "MD" # nve, nvt, npt

    # TODO: !!!!
    # check whether submit jobs
    # check system symbols with type list
    # check lost atoms when collecting

    collection_params = dict(
        selection_tags = ["converged", "traj"]
    )

    def _parse_drivers(self, exp_dict: dict):
        """ create a list of workers based on dyn params
        """
        dyn_params = exp_dict["create"]["driver"]
        #print(dyn_params)

        backend = dyn_params.pop("backend", None)

        # TODO: merge driver's init and run together
        dyn_params = dict(
            **dyn_params.get("init", {}),
            **dyn_params.get("run", {})
        )

        p = MDParams(**dyn_params)
        dcls_list = create_dataclass_from_dict(MDParams, dyn_params)

        drivers = []
        for p in dcls_list:
            p_ = dataclasses.asdict(p)
            run_params_ = dict(steps=p_.pop("steps", 0))
            init_params_ = p_.copy()
            task = init_params_.pop("task")
            p_ = dict(init=init_params_, run=run_params_)
            p_.update(backend=backend)
            p_.update(task=task)

            driver = self.pot_manager.create_driver(p_)
            drivers.append(driver)

        return drivers
    
    def _prior_create(self, input_params: dict, *args, **kwargs):
        """"""
        selector = super()._prior_create(input_params)

        drivers = self._parse_drivers(input_params)

        return drivers, selector
    
    def _single_create(self, res_dpath, frames, cons_text, actions, *args, **kwargs):
        """"""
        # - run over systems
        for i, atoms in enumerate(frames):
            # - set working dir
            #name = atoms.info.get("name", "cand"+str(i))
            name = "cand" + str(i)
            cand_path = self.step_dpath / name

            # - run simulation
            for iw, driver in enumerate(actions):
                driver.directory = cand_path / ("w"+str(iw))
                # TODO: run directly or attach a machine
                driver.run(atoms, constraint=cons_text) # NOTE: other run_params have already been set

        return
    
    def _single_collect(self, res_dpath, frames, cons_text, actions, selector, *args, **kwargs):
        """"""
        # - run over systems

        # NOTE: not save all explored configurations
        #       since they are too many
        traj_dir_groups = {}
        for i, atoms in enumerate(frames):
            # - set working dir
            #name = atoms.info.get("name", "cand"+str(i))
            name = "cand" + str(i)
            cand_path = res_dpath / "create" / name

            # - run simulation
            for iw, driver in enumerate(actions):
                driver_id = "w"+str(iw)
                traj_dir = cand_path / driver_id
                if driver_id in traj_dir_groups:
                    traj_dir_groups[driver_id].append(traj_dir)
                else:
                    traj_dir_groups[driver_id] = [traj_dir]

        # TODO: replace with composition
        traj_period = self.collection_params["traj_period"]
        
        merged_traj_frames = []
        from GDPy.computation.utils import read_trajectories
        for driver_id, traj_dirs in traj_dir_groups.items():
            # print("traj_dirs: ", traj_dirs) # equal number of candidates
            traj_fpath = self.step_dpath / f"traj_frames-{driver_id}.xyz"
            traj_ind_fpath = self.step_dpath / f"traj_indices-{driver_id}.xyz"
            with CustomTimer(name=f"collect-trajectories-{driver_id}"):
                cur_traj_frames = read_trajectories(
                    driver, traj_dirs,
                    traj_period, traj_fpath, traj_ind_fpath
                )
            merged_traj_frames.extend(cur_traj_frames)
        
        # - select
        if selector:
            select_dpath = self._make_step_dir(res_dpath, "select")
            print(select_dpath)

            selector.prefix = "traj"
            selector.directory = select_dpath

            selected_frames = selector.select(merged_traj_frames)

        return
    

if __name__ == '__main__':
    pass