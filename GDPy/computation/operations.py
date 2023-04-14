#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NoReturn

from ase.io import read, write

from GDPy.core.operation import Operation
from GDPy.computation.worker.drive import DriverBasedWorker


class drive(Operation):

    """Drive structures.
    """

    def __init__(
        self, builder, potter, driver, scheduler, 
        read_traj=True, traj_period: int=1,
    ):
        """"""
        super().__init__([builder, potter, driver, scheduler])

        self.read_traj = read_traj
        self.traj_period = traj_period

        return
    
    def forward(self, frames, potter, driver, scheduler):
        """"""
        super().forward()

        driver = potter.create_driver(driver) # use external backend
        worker = DriverBasedWorker(potter, driver, scheduler)
        worker.directory = self.directory

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


if __name__ == "__main__":
    ...