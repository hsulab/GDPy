#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import pathlib
from typing import NoReturn, List
import warnings

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.core.operation import Operation
from GDPy.core.register import registers
from GDPy.data.dataset import XyzDataloader

@registers.operation.register
class end_session(Operation):

    def __init__(self, *args) -> NoReturn:
        super().__init__(args)
    
    def forward(self, *args):
        """"""
        return super().forward()

@registers.operation.register
class chain(Operation):

    """Merge arbitrary nodes' outputs into one list.
    """

    status = "finished" # Always finished since it is not time-consuming

    def __init__(self, nodes, *args, **kwargs) -> NoReturn:
        """"""
        super().__init__(nodes)

        # - some operation parameters

        return
    
    def forward(self, *outputs):
        """"""
        super().forward()
        print("chain outputs: ", outputs)

        return list(itertools.chain(*outputs))

@registers.operation.register
class map(Operation):

    """Give each input node a name and construct a dict.

    This is useful when validating structures from different systems.

    """

    status = "finished"

    def __init__(self, nodes, names, directory="./") -> NoReturn:
        """"""
        super().__init__(nodes, directory)

        assert len(nodes) == len(names), "Numbers of nodes and names are inconsistent."
        self.names = names

        return
    
    def forward(self, *outputs):
        """"""
        super().forward()

        ret = {}
        for k, v in zip(self.names, outputs):
            ret[k] = v
        
        print("map ret: ", ret)

        return ret

@registers.operation.register
class merge_workers(Operation):

    """Merge results of workers.
    """

    def __init__(self, *workers) -> NoReturn:
        """"""
        super().__init__(workers)

        #self.read_traj = read_traj
        #self.traj_period = traj_period

        return
    
    def forward(self, *end_frames_batch):
        """"""
        super().forward()

        prev_confids, curr_confids = [], []

        cur_confid = 0

        all_traj_frames = []
        for drive_node in self.input_nodes:
            worker = drive_node.worker
            cached_trajs_fpath = self.directory/(worker.directory.name+"trajs.xyz")
            if not cached_trajs_fpath.exists():
                trajectories = worker.retrieve(
                    read_traj=True,
                    #read_traj = self.read_traj, 
                    #traj_period = self.traj_period,
                    #include_first = self.include_first, 
                    #include_last = self.include_last,
                    ignore_retrieved=False # TODO: ignore misleads...
                )

                # - save trajs
                flatten_traj_frames = []
                for j, traj in enumerate(trajectories):
                    # -- deal with confid and step to avoid wdir conflicts
                    for atoms in traj:
                        prev_confids.append(atoms.info["confid"])
                        atoms.info["confid"] = cur_confid
                    curr_confids.extend([cur_confid]*len(traj))
                    assert len(prev_confids) == len(curr_confids), "Inconsistent confids."
                    cur_confid += 1
                    # -- save trajs
                    flatten_traj_frames.extend(traj)
                write(cached_trajs_fpath, flatten_traj_frames)

                np.savetxt(
                    self.directory/"confid_map.txt", np.array([prev_confids,curr_confids]).T,
                    fmt="%12d", header="{:>11s}{:>12s}".format("prev", "curr")
                )

                # - sort atoms info...
                # -- add source to atoms...
                for atoms in flatten_traj_frames:
                    atoms.info["drive_source"] = str(worker.directory)
            else:
                flatten_traj_frames = read(cached_trajs_fpath, ":")
            self.pfunc(f"{str(worker.directory)} nframes: {len(flatten_traj_frames)}")
            
            all_traj_frames.extend(flatten_traj_frames)
        
        self.pfunc(f"nframes: {len(all_traj_frames)}")

        return all_traj_frames

@registers.operation.register
class transfer(Operation):

    """Transfer worker results to target destination.
    """

    def __init__(self, structure, target_dir, version, system="mixed", directory="./") -> NoReturn:
        """"""
        input_nodes = [structure]
        super().__init__(input_nodes=input_nodes, directory=directory)

        self.target_dir = pathlib.Path(target_dir).resolve()
        self.version = version

        self.system = system # molecule/cluster, surface, bulk

        return
    
    def forward(self, frames: List[Atoms]):
        """"""
        super().forward()

        self.pfunc(f"target dir: {str(self.target_dir)}")

        # - check chemical symbols
        system_dict = {} # {formula: [indices]}

        formulae = [a.get_chemical_formula() for a in frames]
        for k, v in itertools.groupby(enumerate(formulae), key=lambda x: x[1]):
            system_dict[k] = [x[0] for x in v]
        
        # - transfer data
        for formula, curr_indices in system_dict.items():
            # -- TODO: check system type
            system_type = self.system # currently, use user input one
            # -- name = description+formula+system_type
            dirname = "-".join([self.directory.parent.name, formula, system_type])
            target_subdir = self.target_dir/dirname
            target_subdir.mkdir(parents=True, exist_ok=True)

            # -- save frames
            curr_frames = [frames[i] for i in curr_indices]
            curr_nframes = len(curr_frames)

            strname = self.version + ".xyz"
            target_destination = self.target_dir/dirname/strname
            if not target_destination.exists():
                write(target_destination, curr_frames)
                self.pfunc(f"nframes {curr_nframes} -> {target_destination.name}")
            else:
                warnings.warn(f"{target_destination} exists.", UserWarning)
        
        dataset = XyzDataloader(self.target_dir)
        self.status = "finished"

        return dataset

if __name__ == "__main__":
    ...