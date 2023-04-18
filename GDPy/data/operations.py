#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NoReturn

import numpy as np

from ase.io import read, write

from GDPy.core.operation import Operation

class merge(Operation):

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
            print(worker.directory)
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
                    fmt="%12d", header="{:>11d}{:>12d}".format("prev", "curr")
                )

                # - sort atoms info...
                # -- add source to atoms...
                for atoms in flatten_traj_frames:
                    atoms.info["drive_source"] = str(worker.directory)
            else:
                flatten_traj_frames = read(cached_trajs_fpath, ":")
            

            all_traj_frames.extend(flatten_traj_frames)
        
        print("nframes: ", len(all_traj_frames))

        return all_traj_frames

if __name__ == "__main__":
    ...