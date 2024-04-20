#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import time
from typing import Optional, List, Mapping

import omegaconf

from ase import Atoms

from ..core.register import registers
from ..core.variable import Variable
from ..core.operation import Operation
from ..data.array import AtomsNDArray


@registers.operation.register
class pair_stru(Operation):

    def __init__(self, structures, method="couple", pairs=None, directory="./") -> None:
        """"""
        super().__init__(input_nodes=[structures], directory=directory)

        assert method in [
            "couple",
            "concat",
            "custom",
        ], "pair method should be either couple or concat."
        self.method = method
        self.pairs = pairs

        return

    def forward(self, structures: AtomsNDArray) -> List[List[Atoms]]:
        """"""
        super().forward()

        # NOTE: assume this is a 1-D array
        self._debug(f"structures: {structures}")
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

        # Must update status at the end of forward! Otherwise, the status will be 
        # overwritten in active session by node.reset().
        self.status = "finished"

        return AtomsNDArray(pair_structures)


@registers.operation.register
class react(Operation):

    def __init__(
        self, structures, reactor, batchsize: Optional[int] = None, directory="./"
    ) -> None:
        """"""
        super().__init__(input_nodes=[structures, reactor], directory=directory)

        self.batchsize = batchsize

        return

    def forward(self, structures: AtomsNDArray, reactors):
        """"""
        super().forward()

        # - assume structures be an AtomsNDArray with a shape of (num_pairs, 2)
        # structures = [[x[-1] for x in s] for s in structures]
        # if isinstance(structures, list):
        #     # - from list_nodes operation
        #     structures = AtomsNDArray([x.tolist() for x in structures])
        # else:
        #     # - from pair operation
        #     structures = AtomsNDArray(structures)
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
            flag_fpath = reactor.directory / "FINISHED"
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
