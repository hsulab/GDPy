#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import time
from typing import Optional, List, Mapping

from ase import Atoms

from ..core.register import registers
from ..core.variable import Variable
from ..core.operation import Operation
from ..data.array import AtomsNDArray
from ..worker.react import ReactorBasedWorker


@registers.variable.register
class ReactorVariable(Variable):

    """Create a ReactorBasedWorker.

    TODO:
        Broadcast driver params to give several workers?

    """

    def __init__(self, potter, driver: dict, scheduler={}, batchsize=1, directory="./", *args, **kwargs):
        """"""
        # - save state by all nodes
        self.potter = self._load_potter(potter)
        self.driver = self._load_driver(driver)
        self.scheduler = self._load_scheduler(scheduler)

        self.batchsize = batchsize

        # - create a reactor
        #reactor = self.potter.create_reactor(kwargs)
        workers = self._create_workers(self.potter, self.driver, self.scheduler, self.batchsize)

        super().__init__(initial_value=workers, directory=directory)

        return

    def _load_potter(self, inp):
        """"""
        potter = None
        if isinstance(inp, Variable):
            potter = inp.value
        elif isinstance(inp, dict):
            potter_params = copy.deepcopy(inp)
            name = potter_params.get("name", None)
            potter = registers.create(
                "manager", name, convert_name=True,
            )
            potter.register_calculator(potter_params.get("params", {}))
            potter.version = potter_params.get("version", "unknown")
        else:
            raise RuntimeError(f"Unknown {inp} for the potter.")

        return potter

    def _load_driver(self, inp) -> List[dict]:
        """Load drivers from a Variable or a dict."""
        #print("driver: ", inp)
        drivers = [] # params
        if isinstance(inp, Variable):
            drivers = inp.value
        elif isinstance(inp, dict): # assume it only contains one driver
            driver_params = copy.deepcopy(inp)
            #driver = self.potter.create_driver(driver_params) # use external backend
            drivers = [driver_params]
        else:
            raise RuntimeError(f"Unknown {inp} for drivers.")

        return drivers

    def _load_scheduler(self, inp):
        """"""
        scheduler = None
        if isinstance(inp, Variable):
            scheduler = inp.value
        elif isinstance(inp, dict):
            scheduler_params = copy.deepcopy(inp)
            backend = scheduler_params.pop("backend", "local")
            scheduler = registers.create(
                "scheduler", backend, convert_name=True, **scheduler_params
            )
        else:
            raise RuntimeError(f"Unknown {inp} for the scheduler.")

        return scheduler
    
    def _create_workers(self, potter, drivers: List[dict], scheduler, batchsize: int=1, *args, **kwargs):
        """"""
        workers = []
        for driver_params in drivers:
            driver = potter.create_reactor(driver_params)
            worker = ReactorBasedWorker(potter, driver, scheduler)
            workers.append(worker)
        
        for worker in workers:
            worker.batchsize = batchsize

        return workers


@registers.operation.register
class pair_stru(Operation):

    status: str = "finished"

    def __init__(self, structures, method="couple", pairs=None, directory="./") -> None:
        """"""
        super().__init__(input_nodes=[structures], directory=directory)

        assert method in ["couple", "concat", "custom"], "pair method should be either couple or concat."
        self.method = method
        self.pairs = pairs

        return
    
    def forward(self, structures: AtomsNDArray) -> List[List[Atoms]]:
        """"""
        super().forward()

        # NOTE: assume this is a 1-D array
        print(f"structures: {structures}")
        intermediates = structures.get_marked_structures()

        nframes = len(intermediates)
        rankings = list(range(nframes))

        if self.method == "couple":
            pair_indices = [rankings[0::2], rankings[1::2]]
        elif self.method == "concat":
            pair_indices = [rankings[:-1], rankings[1:]]
        elif self.method == "custom":
            pair_indices = self.pairs
        else:
            ...

        pair_structures = []
        for p in pair_indices:
            pair_structures.append([intermediates[i] for i in p])

        return AtomsNDArray(pair_structures)


@registers.operation.register
class react(Operation):

    def __init__(self, structures, reactor, batchsize: Optional[int]=None, directory="./") -> None:
        """"""
        super().__init__(input_nodes=[structures, reactor], directory=directory)

        self.batchsize = batchsize

        return
    
    def forward(self, structures: List[AtomsNDArray], reactors):
        """"""
        super().forward()

        if isinstance(structures, list):
            # - from list_nodes operation
            structures = AtomsNDArray([x.tolist() for x in structures])
        else:
            # - from pair operation
            structures = AtomsNDArray(structures)
        print(f"structures: {structures}")
        print(f"structures: {structures[0]}")

        # - assume structures contain a List of trajectory/frames pair
        #   take the last frame out since it is minimised?
        #structures = [[x[-1] for x in s] for s in structures]
        nreactions = len(structures)

        # - create reactors
        for i, reactor in enumerate(reactors):
            reactor.directory = self.directory / f"r{i}"
            if self.batchsize is not None:
                reactor.batchsize = self.batchsize
            else:
                reactor.batchsize = nreactions
        nreactors = len(reactors)

        if nreactors == 1:
            reactors[0].directory = self.directory

        # - align structures
        reactor_status = []
        for i, reactor in enumerate(reactors):
            flag_fpath = reactor.directory/"FINISHED"
            self._print(f"run reactor {i} for {nreactions} nframes")
            if not flag_fpath.exists():
                reactor.run(structures)
                reactor.inspect(resubmit=True)
                if reactor.get_number_of_running_jobs() == 0:
                    with open(flag_fpath, "w") as fopen:
                        fopen.write(
                            f"FINISHED AT {time.asctime( time.localtime(time.time()) )}."
                        )
                    reactor_status.append(True)
                else:
                    reactor_status.append(False)
            else:
                with open(flag_fpath, "r") as fopen:
                    content = fopen.readlines()
                self._print(content)
                reactor_status.append(True)
        
        if all(reactor_status):
            self.status = "finished"

        return reactors


if __name__ == "__main__":
    ...