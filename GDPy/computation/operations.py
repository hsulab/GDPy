#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NoReturn, List

from ase import Atoms
from ase.io import read, write

from GDPy.core.operation import Operation
from GDPy.core.register import registers
from GDPy.computation.worker.drive import DriverBasedWorker

@registers.operation.register
class work(Operation):

    def __init__(self, potter, driver, scheduler, batchsize: int=None) -> NoReturn:
        """"""
        super().__init__([potter,driver,scheduler])

        self.batchsize = batchsize

        return
    
    def forward(self, potter, driver_params, scheduler):
        """"""
        super().forward()

        driver = potter.create_driver(driver_params) # use external backend
        worker = DriverBasedWorker(potter, driver, scheduler)
        worker.directory = self.directory

        return worker

@registers.operation.register
class drive(Operation):

    """Drive structures.
    """

    def __init__(
        self, builder, potter, driver, scheduler, 
        batchsize: int=None, read_traj=True, traj_period: int=1,
    ):
        """"""
        super().__init__([builder, potter, driver, scheduler])

        self.batchsize = batchsize
        self.read_traj = read_traj
        self.traj_period = traj_period

        return
    
    def forward(self, frames, potter, driver, scheduler):
        """"""
        super().forward()

        driver = potter.create_driver(driver) # use external backend
        worker = DriverBasedWorker(potter, driver, scheduler)
        worker.directory = self.directory

        if self.batchsize is not None:
            worker.batchsize = self.batchsize
        else:
            nframes = len(frames)
            worker.batchsize = nframes

        self.worker = worker

        new_frames = []

        cached_fpath = self.directory/"output.xyz"
        if not cached_fpath.exists():
            worker.run(frames)
            worker.inspect(resubmit=True)
            # TODO: check whether finished this operation...
            if worker.get_number_of_running_jobs() == 0:
                trajectories = worker.retrieve(
                    read_traj = self.read_traj, traj_period = self.traj_period,
                    include_first = True, include_last = True,
                    ignore_retrieved=False # TODO: ignore misleads...
                )
                for traj in trajectories:
                    new_frames.append(traj[-1])
                write(cached_fpath, new_frames)
        else:
            new_frames = read(cached_fpath, ":")

        return new_frames

class extract_trajectories(Operation):

    """Extract dynamics trajectories from a drive-node's worker.
    """

    def __init__(
        self, drive_node, read_traj=True, traj_period: int=1,
        include_first=True, include_last=True
    ) -> NoReturn:
        """"""
        super().__init__([drive_node])

        self.read_traj = read_traj
        self.traj_period = traj_period

        self.include_first = include_first
        self.include_last = include_last

        return
    
    def forward(self, frames: List[Atoms]):
        """
        Args:
            frames: Structures from drive-node's output (end frame of each traj).
            
        """
        super().forward()

        # - save ALL end frames
        cached_conv_fpath = self.directory/"last.xyz" # converged frames
        if not cached_conv_fpath.exists():
            write(cached_conv_fpath, frames)

        # - extract trajetories
        cached_trajs_fpath = self.directory/"trajs.xyz"
        if not cached_trajs_fpath.exists():
            worker = self.input_nodes[0].worker

            trajectories = worker.retrieve(
                read_traj = self.read_traj, traj_period = self.traj_period,
                include_first = self.include_first, include_last = self.include_last,
                ignore_retrieved=False # TODO: ignore misleads...
            )
            flatten_traj_frames = []
            for traj in trajectories:
                flatten_traj_frames.extend(traj)
            write(cached_trajs_fpath, flatten_traj_frames)
        else:
            flatten_traj_frames = read(cached_trajs_fpath, ":")
        
        # TODO: reconstruct trajs to List[List[Atoms]]

        return flatten_traj_frames


if __name__ == "__main__":
    ...