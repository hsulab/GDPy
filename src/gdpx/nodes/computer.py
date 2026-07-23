#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from typing import Optional

import omegaconf
from gdpx.core.register import registers
from gdpx.factory.computer import create_workers
from gdpx.session.variable import Variable
from gdpx.utils.parser import parse_input_file


@registers.variable.register
class ComputerChainVariable(Variable):

    def __init__(self, computers, directory=pathlib.Path.cwd()):
        """"""
        value = self._canonicalise_input_nodes([computers])
        super().__init__(value)

        self._init_params = copy.deepcopy([[w.as_dict() for w in c] for c in value])

        return

    def _canonicalise_input_nodes(self, input_nodes):
        """"""
        (computers,) = input_nodes

        if isinstance(computers, list) or isinstance(computers, omegaconf.ListConfig):
            computers_ = []
            for computer in computers:
                if isinstance(computer, str) or isinstance(computer, pathlib.Path):
                    computer = parse_input_file(input_fpath=computer)
                computer_ = None
                if isinstance(computer, dict) or isinstance(computer, omegaconf.DictConfig):
                    computer_ = ComputerVariable(**computer)
                elif isinstance(computer, ComputerVariable):
                    computer_ = computer
                else:
                    raise RuntimeError(f"Unknown input for computer with a type of {computer}.")
                computers_.append(computer_)
            computers = computers_
        else:
            raise Exception()

        # TODO: We'd better make computer_chain as a class?
        num_workers_per_chain = len(computers[0].value)

        # Add the first chain
        value = [[] for _ in range(num_workers_per_chain)]  # size (num_chainsteps, num_chains)
        for i, computer in enumerate(computers):  # i -> chainstep index
            num = len(computer.value)
            if num != num_workers_per_chain:
                raise Exception(f"chainstep.{i:>02d} need {num_workers_per_chain:>2d} workers but got {num:>2d}.")
            for j, worker in enumerate(computer.value):  # j -> chain index
                value[j].append(worker)

        return value

    def as_dict(self) -> dict:
        """"""

        return self._init_params


@registers.variable.register
class ComputerVariable(Variable):

    def __init__(
        self,
        potter,
        driver={},
        scheduler={},
        *,
        use_grid: bool = False,
        estimate_uncertainty: Optional[bool] = None,
        switch_backend: Optional[str] = None,
        batchsize: int = 1,
        share_wdir: bool = False,
        use_single: bool = False,
        retain_info: bool = False,
        directory = "./",
    ):
        """"""
        # Workflow references are unwrapped at the adapter boundary; the
        # underlying factory deliberately knows nothing about Variable.
        potter_value = potter.value if isinstance(potter, Variable) else potter
        driver_value = driver.value if isinstance(driver, Variable) else driver
        scheduler_value = scheduler.value if isinstance(scheduler, Variable) else scheduler

        # Save input parameters
        self._init_params = copy.deepcopy(
            dict(
                potter=potter,
                driver=driver,
                scheduler=scheduler,
                use_grid=use_grid,
                estimate_uncertainty=estimate_uncertainty,
                switch_backend=switch_backend,
                batchsize=batchsize,
                share_wdir=share_wdir,
                use_single=use_single,
                retain_info=retain_info,
            )
        )

        workers = create_workers(
            dict(
                potter=potter_value,
                driver=driver_value,
                scheduler=scheduler_value,
                estimate_uncertainty=estimate_uncertainty,
                switch_backend=switch_backend,
                batchsize=batchsize,
                share_wdir=share_wdir,
                use_single=use_single,
                retain_info=retain_info,
            ),
            directory=directory,
            print_func=self._print,
            use_grid=use_grid,
        )
        super().__init__(workers, directory=directory)

        self.use_single = use_single

        return

    def as_dict(self) -> dict:
        """"""

        return self._init_params


if __name__ == "__main__":
    ...
