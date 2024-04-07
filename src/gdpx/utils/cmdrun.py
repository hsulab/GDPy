#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import subprocess

from ase.calculators.calculator import EnvironmentError, CalculationFailed


def run_ase_calculator(name: str, command, directory):
    """Run vasp from the command.

    ASE Vasp does not treat restart of a MD simulation well. Therefore, we run
    directly from the command if some input files aready exist.

    For example, we use existed INCAR for VASP.

    """
    try:
        proc = subprocess.Popen(command, shell=True, cwd=directory)
    except OSError as err:
        # Actually this may never happen with shell=True, since
        # probably the shell launches successfully.  But we soon want
        # to allow calling the subprocess directly, and then this
        # distinction (failed to launch vs failed to run) is useful.
        msg = f"Failed to execute `{command}`"
        raise EnvironmentError(msg) from err

    errorcode = proc.wait()

    if errorcode:
        path = os.path.abspath(directory)
        msg = (
            f"Calculator `{name}` failed with command `{command}` "
            + f"failed in `{path}` with error code `{errorcode}`"
        )
        raise CalculationFailed(msg)

    return


if __name__ == "__main__":
    ...
