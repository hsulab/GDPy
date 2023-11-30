#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy 
from typing import List

from .. import registers, Variable, SingleWorker

from ..expedition import AbstractExpedition
from ..mc.mc import MonteCarlo


class BasinHoppingVariable(Variable):

    def __init__(self, builder, worker, directory="./", *args, **kwargs) -> None:
        """"""
        # - builder
        if isinstance(builder, dict):
            builder_params = copy.deepcopy(builder)
            builder_method = builder_params.pop("method")
            builder = registers.create(
                "builder", builder_method, convert_name=False, **builder_params
            )
        else: # variable
            builder = builder.value
        # - worker
        if isinstance(worker, dict):
            worker_params = copy.deepcopy(worker)
            worker = registers.create("variable", "computer", convert_name=True, **worker_params).value[0]
        elif isinstance(worker, Variable): # computer variable
            worker = worker.value[0]
        elif isinstance(worker, SingleWorker): # assume it is a DriverBasedWorker
            worker = worker
        else:
            raise RuntimeError(f"BasinHopping needs a SingleWorker instead of a {worker}")
        engine = self._create_engine(builder, worker, *args, **kwargs)
        engine.directory = directory
        super().__init__(initial_value=engine, directory=directory)

        return
    
    def _create_engine(self, builder, worker, *args, **kwargs) -> None:
        """"""
        engine = BasinHopping(builder, worker, *args, **kwargs)

        return engine


class BasinHopping(MonteCarlo):

    def __init__(
        self, builder: dict, worker: dict, operators: List[dict], 
        convergence: dict, random_seed=None, dump_period: int=1, ckpt_period: int=100,
        restart: bool = False, directory="./", *args, **kwargs
    ) -> None:
        """"""
        super().__init__(
            builder, worker, operators, convergence, random_seed, dump_period, 
            restart, directory, *args, **kwargs
        )

        self.ckpt_period = ckpt_period

        return


if __name__ == "__main__":
    ...