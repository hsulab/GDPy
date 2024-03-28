#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union

import omegaconf

from . import registers
from ..core.variable import Variable, DummyVariable
from ..core.operation import Operation


class DescriberVariable(Variable):

    def __init__(self, directory="./", *args, **kwargs):
        """"""
        name = kwargs.pop("name", "soap")
        describer = registers.create("describer", name, convert_name=False, **kwargs)

        super().__init__(initial_value=describer, directory=directory)

        return


class describe(Operation):

    def __init__(
        self,
        structures,
        describer,
        worker=DummyVariable(),
        directory: Union[str, pathlib.Path] = "./",
    ) -> None:
        """"""
        super().__init__(
            input_nodes=[structures, describer, worker], directory=directory
        )

        return

    def _preprocess_input_nodes(self, input_nodes):
        """"""
        structures, describer, worker = input_nodes

        if isinstance(describer, dict) or isinstance(
            describer, omegaconf.dictconfig.DictConfig
        ):
            describer = DescriberVariable(
                directory=self.directory / "describer", **describer
            )

        return structures, describer, worker

    def forward(self, structures, describer, workers):
        """"""
        super().forward()

        # - verify the worker
        if workers is not None:
            nworkers = len(workers)
            assert (
                nworkers == 1
            ), f"{self.__class__.__name__} only accepts one worker but {nworkers} were given."
            worker = workers[0]
            worker.directory = self.directory/"worker"
        else:
            worker = None
        
        # - dataset?

        # - compute descriptors...
        describer.directory = self.directory
        status = describer.run(structures, worker, )

        #self.status = "finished"
        self.status = status

        return structures


if __name__ == "__main__":
    ...
