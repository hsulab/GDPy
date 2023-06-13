#!/usr/bin/env python3
# -*- coding: utf-8 -*

import pathlib
import time

from ..utils.command import parse_input_file
from ..core.operation import Operation
from ..core.register import registers
from ..worker.explore import RoutineBasedWorker


"""
"""

class routine(Operation):

    def __init__(self, routine, scheduler, directory="./") -> None:
        """"""
        input_nodes = [routine, scheduler]
        super().__init__(input_nodes, directory)

        return

    def forward(self, routine, scheduler):
        """Perform a routine and forward results for further analysis.

        Returns:
            Workers that store structures.
        
        """
        super().forward()

        # - 
        basic_workers = []
        
        # - run routine with a worker
        worker = RoutineBasedWorker(routine, scheduler)
        worker.directory = self.directory

        worker.run()
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            basic_workers = worker.retrieve(include_retrieved=True)
            # print("basic_workers: ", basic_workers)
            # for w in basic_workers:
            #     print(w.directory)
            self.status = "finished"
        else:
            ...

        return basic_workers


def run_routine(config_params: dict, wait: float=None, directory="./"):
    """"""
    directory = pathlib.Path("./")

    method = config_params.pop("method")
    routine = registers.create("variable", method, convert_name=True, **config_params).value

    routine.directory = directory

    if wait is not None:
        for i in range(1000):
            routine.run()
            if routine.read_convergence():
                break
            time.sleep(wait)
            print(f"wait {wait} seconds...")
        else:
            ...
    else:
        routine.run()
        ...

    return


if __name__ == "__main__":
    ...