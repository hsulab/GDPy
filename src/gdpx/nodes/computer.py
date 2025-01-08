#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Any, Optional

from gdpx.core.register import registers
from gdpx.worker.drive import DriverBasedWorker
from gdpx.worker.single import SingleWorker
from gdpx.worker.interface import ComputerVariable


def canonicalise_worker(inp_worker: Optional[Any]) -> Optional[DriverBasedWorker]:
    """Canonicalise the worker."""
    worker = None
    if isinstance(inp_worker, dict):
        worker_params = copy.deepcopy(inp_worker)
        worker = registers.create(
            "variable", "computer", convert_name=True, **worker_params
        ).value[0]
    elif isinstance(inp_worker, list):  # assume it is from a computervariable
        worker = inp_worker[0]
    elif isinstance(inp_worker, ComputerVariable):
        worker = inp_worker.value[0]
    elif isinstance(inp_worker, DriverBasedWorker) or isinstance(
        inp_worker, SingleWorker
    ):
        worker = inp_worker
    else:
        if inp_worker is None:
            worker = None
        else:
            raise RuntimeError(f"Unknown worker type {inp_worker}.")

    return worker


if __name__ == "__main__":
    ...
  
