#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import NoReturn, List

from ase import Atoms
from ase.io import read, write

from gdpx.core.variable import Variable
from gdpx.core.operation import Operation
from gdpx.core.register import registers


@registers.variable.register
class DriverVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        # - compat
        copied_params = copy.deepcopy(kwargs)
        merged_params = dict(
            task = copied_params.get("task", "min"),
            backend = copied_params.get("backend", "external"),
            ignore_convergence = copied_params.get("ignore_convergence", False)
        )
        merged_params.update(**copied_params.get("init", {}))
        merged_params.update(**copied_params.get("run", {}))

        initial_value = self._broadcast_drivers(merged_params)

        super().__init__(initial_value)

        return
    
    def _broadcast_drivers(self, params: dict) -> List[dict]:
        """Broadcast parameters if there were any parameter is a list."""
        # - find longest params
        plengths = []
        for k, v in params.items():
            if isinstance(v, list):
                n = len(v)
            else: # int, float, string
                n = 1
            plengths.append((k,n))
        plengths = sorted(plengths, key=lambda x:x[1])
        # NOTE: check only has one list params
        assert sum([p[1] > 1 for p in plengths]) <= 1, "only accept one param as list."

        # - convert to dataclass
        params_list = []
        maxname, maxlength = plengths[-1]
        for i in range(maxlength):
            curr_params = {}
            for k, n in plengths:
                if n > 1:
                    v = params[k][i]
                else:
                    v = params[k]
                curr_params[k] = v
            params_list.append(curr_params)

        return params_list

if __name__ == "__main__":
    ...