#!/usr/bin/env python3
# -*- coding: utf-8 -*


"""Grid computation worker.

This module provides a thin wrapper around :class:`DriverBasedWorker`
that configures it for multiple (potter, driver) configurations.

The functionality has been unified into :class:`~gdpx.worker.drive.DriverBasedWorker`;
this module exists for backward compatibility.
"""


import json
import subprocess


from gdpx.computation.driver import BaseDriver
from gdpx.core.register import registers
from gdpx.potential.manager import BasePotentialManager
from gdpx.scheduler.local import LocalScheduler
from gdpx.scheduler.scheduler import BaseScheduler

from .drive import DriverBasedWorker
from .pairing import Pairing


def run_command(directory, command, comment="", timeout=None, print_func=print):
    """Run a shell command in *directory*."""
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if timeout is None:
        errorcode = proc.wait()
    else:
        errorcode = proc.wait(timeout=timeout)

    output = proc.stdout
    if output is not None:
        msg = "Message: " + "".join(output.readlines())
    else:
        msg = ""
    print_func(msg)

    if errorcode:
        raise RuntimeError("Error in %s at %s." % (comment, directory))

    return msg


STRU_ID_KEY: str = "identifier"
BATCH_ID_KEY: str = "gdir"


@registers.worker.register
class GridDriverBasedWorker(DriverBasedWorker):
    """Grid of potters × drivers.

    .. deprecated::
        Use :class:`DriverBasedWorker` with multiple drivers and
        ``pairing=Pairing.REPEAT`` (or ``Pairing.PARTITION``) instead.
    """

    def __init__(
        self,
        potters: list[BasePotentialManager],
        drivers: list[BaseDriver],
        scheduler: BaseScheduler = LocalScheduler(),
        directory="./",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            scheduler=scheduler,
            directory=directory,
            batchsize=kwargs.pop("batchsize", 1),
            pairing=Pairing.AUTO,
            *args, **kwargs,
        )
        self.set_drivers(*drivers)
        # Store potters for serialisation
        self._grid_potters = potters

    def _write_inputs(self, identifier: str, batch_numbers, wdir_names, has_broadcast: bool = False):
        inp_fpath = self.directory / "_data" / f"inp-{identifier}.json"
        if inp_fpath.exists():
            return

        grid_params = {}
        grid_params["grid"] = []

        for i, (ib, wdir_name, potter) in enumerate(
            zip(batch_numbers, wdir_names, self._grid_potters)
        ):
            if not has_broadcast:
                stru_i = i
            else:
                stru_i = 0
            comput_data = {"batch": ib, "wdir_name": wdir_name}
            comput_data["builder"] = dict(
                method="reader",
                fname=str((self.directory / "_data" / f"{identifier}.xyz").relative_to(self.directory)),
                index=f"{stru_i}",
            )
            comput_data["computer"] = {}
            comput_data["computer"]["potter"] = potter.as_dict()
            comput_data["computer"]["driver"] = self.drivers[i].as_dict()
            grid_params["grid"].append(comput_data)

        with open(inp_fpath, "w") as fopen:
            json.dump(grid_params, fopen, indent=2)

    def as_dict(self) -> dict:
        params = super().as_dict()
        return params


if __name__ == "__main__":
    ...
