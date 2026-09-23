#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Any, Optional

import numpy as np
from ase import Atoms
from ase.io import read, write

from gdpx.validator.validator import BaseValidator
from gdpx.worker.drive import DriverBasedWorker


def run_worker(
    working_directory: pathlib.Path, frames: list[Atoms], worker: DriverBasedWorker
) -> tuple[Optional[list[Atoms]], Optional[list[Atoms]]]:
    """"""
    assert isinstance(worker, DriverBasedWorker), "Worker must be a DriverBasedWorker."

    cache_fpath = working_directory / "calc.xyz"
    if cache_fpath.exists():
        ini_frames = read(working_directory / "calc_ini.xyz", ":")
        end_frames = read(cache_fpath, ":")
        return ini_frames, end_frames

    _ = worker.run(frames)
    _ = worker.inspect(resubmit=True)
    if worker.get_number_of_running_jobs() == 0:
        trajectories = worker.retrieve(include_retrieved=True, use_archive=True)
        ini_frames = [t[0] for t in trajectories]
        write(working_directory / "calc_ini.xyz", ini_frames)
        end_frames = [t[-1] for t in trajectories]
        write(cache_fpath, end_frames)
    else:
        ini_frames, end_frames = None, None

    return ini_frames, end_frames  # type: ignore


def get_uncertainty_histogram(frames: list[Atoms], prop_name: str) -> str:
    """Write uncertainty histogram."""
    properties = np.array([atoms.info.get(prop_name, None) for atoms in frames])

    hist, bin_edges = np.histogram(properties, bins=10)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    content = f"#{prop_name:>11s}  {'count':>12s}\n"
    for centre, count in zip(bin_centres, hist):
        content += f"{centre:>12.4f}  {count:>12d}\n"

    return content


class UncertaintyValidator(BaseValidator):
    """This class validates the uncertainty of structures."""

    name: str = "uncertainty"

    def __init__(self, *args, **kwargs):
        """Initialise the UncertaintyValidator."""
        super().__init__(*args, **kwargs)

        return

    def run(self, structures: Optional[Any], worker: Optional[DriverBasedWorker]) -> bool:
        """Validate the uncertainty of a structure."""
        super().run()

        # Check worker that must have uncertainty estimation
        if worker is not None:
            self._print("Use the worker at run time.")
        else:
            worker = self.worker

        if worker is None:
            raise Exception(
                "The uncertainty validator requires a worker that runs single-point calculation with uncertainty estimation."
            )

        is_uncertainty_enabled = worker.potter.calc_params.get("estimate_uncertainty", False)
        if not is_uncertainty_enabled:
            raise Exception(f"The potential `{worker.potter}` does not support uncertainty estimation.")

        # TODO: We need check if the worker is spc as well.

        # Check what input structures we have
        if structures is not None:
            structures = structures  # => {"reference": AtomsNDArray}
            self._print("Use the structures at run time.")
        else:
            ...

        # Run the calculation
        is_finished = False

        ini_frames, end_frames = run_worker(
            working_directory=self.directory, frames=structures["reference"], worker=worker
        )

        if ini_frames is not None and end_frames is not None:
            is_finished = True
            self._print("Uncertainty validation finished successfully.")
            content = get_uncertainty_histogram(end_frames, "max_devi_f")
            with open(self.directory / "hist.dat", "w") as f:
                f.write(content)
            for line in content.split("\n"):
                self._print(line)
        else:
            is_finished = False

        return is_finished


if __name__ == "__main__":
    ...
