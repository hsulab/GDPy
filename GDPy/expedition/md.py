#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Counter, Union, List

import dataclasses

from ase import Atoms
from ase.io import read, write

from GDPy.expedition.abstract import AbstractExpedition


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


class MDBasedExpedition(AbstractExpedition):

    """
    Exploration Strategies
        brute-force molecular dynamics
        biased molecular dynamics
    
    Initial Systems
        initial structures must be manually prepared
    
    Units
        fs, eV, eV/AA
    """

    name = "MD" # nve, nvt, npt

    # TODO: !!!!
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

            driver = self.pot_worker.potter.create_driver(p_)
            drivers.append(driver)

        return drivers
    
    def _prior_create(self, input_params: dict, *args, **kwargs):
        """"""
        actions = super()._prior_create(input_params)

        drivers = self._parse_drivers(input_params)
        actions["driver"] = drivers

        return actions
    
    def _single_create(self, res_dpath, actions, *args, **kwargs):
        """"""
        generator = actions["generator"]
        self.logger.info(generator.__class__.__name__)
        frames = generator.run(kwargs.get("ran_size", 1))
        self.logger.info(f"number of initial structures: {len(frames)}")
        from GDPy.builder.direct import DirectGenerator
        actions["generator"] = DirectGenerator(frames, res_dpath/"init")

        # - run over systems
        drivers = actions["driver"]

        worker = self.pot_worker
        for iw, driver in enumerate(drivers):
            worker.logger = self.logger
            worker.directory = self.step_dpath/f"w{iw}"
            worker.driver = driver
            worker.batchsize = len(frames)
            worker.run(frames)
        
        # - check if finished
        is_finished = False
        for iw, driver in enumerate(drivers):
            worker.logger = self.logger
            worker.directory = self.step_dpath/f"w{iw}"
            worker.driver = driver
            worker.batchsize = len(frames)
            worker.inspect()
            if worker.get_number_of_running_jobs() > 0:
                is_finished = False
                break
        else:
            is_finished = True

        return is_finished
    
    def _single_collect(self, res_dpath, actions, *args, **kwargs):
        """"""
        generator = actions["generator"]
        self.logger.info(generator.__class__.__name__)
        frames = generator.run(kwargs.get("ran_size", 1))
        self.logger.info(f"number of initial structures: {len(frames)}")

        traj_period = self.collection_params["traj_period"]

        # NOTE: not save all explored configurations
        #       since they are too many

        is_collected = True
        worker = self.pot_worker

        drivers = actions["driver"]
        for iw, driver in enumerate(drivers):
            worker.logger = self.logger
            worker.directory = res_dpath/"create"/f"w{iw}"
            worker.driver = driver
            worker.batchsize = len(frames)
            traj_fpath = self.step_dpath / f"traj_frames-w{iw}.xyz"
            new_frames = worker.retrieve(
                read_traj=True, traj_period=traj_period, include_first=False
            )
            if new_frames:
                write(traj_fpath, new_frames, append=True)
            if len(worker._get_unretrieved_jobs()) > 0:
                is_collected = False
                continue
            # TODO: move this info to worker
            self.logger.info(f"worker {iw} retrieves {len(new_frames)} structures...")

        merged_traj_frames = []
        for i in range(len(drivers)):
            traj_fpath = self.step_dpath / f"traj_frames-w{i}.xyz"
            traj_frames = read(traj_fpath, ":")
            merged_traj_frames.extend(traj_frames)
        self.logger.info(f"total nframes: {len(merged_traj_frames)}")

        # - select
        is_selected = self._single_select(res_dpath, merged_traj_frames, actions)

        return (is_collected and is_selected)
    

if __name__ == "__main__":
    pass