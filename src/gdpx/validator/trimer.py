#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
import pathlib
import warnings

from typing import List

import numpy as np

import matplotlib.pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    ...

from ase import Atoms
from ase.io import read, write

from ..worker.drive import DriverBasedWorker
from .validator import AbstractValidator
from .utils import get_properties


class TrimerValidator(AbstractValidator):

    def __init__(self, angle: List[int], *args, **kwargs):
        """"""
        super().run(*args, **kwargs)
        self.angle = angle

        return

    def run(self, dataset, worker, *args, **kwargs):
        """"""
        super().run()
        self._print(dataset["reference"])
        for prefix, frames in dataset["reference"].items():
            self._irun(prefix, frames, None, worker)

        return

    def _irun(
        self, prefix: str, ref_frames: List[Atoms], pred_frames: List[Atoms], worker
    ):
        """"""
        # - check if input frames are dimers...
        chemicals = []
        for atoms in ref_frames:
            natoms = len(atoms)
            assert natoms == 3, f"Input structure at {self.directory} must be a trimer."
            chemicals.append(atoms.get_chemical_formula())
            assert (
                len(set(chemicals)) == 1
            ), f"Input dimers are different. Maybe {chemicals[0]}?"

        # - read reference
        ref_symbols, ref_energies, ref_forces = get_properties(ref_frames)
        nframes = len(ref_frames)
        ref_natoms = [len(a) for a in ref_frames]

        if pred_frames is None:
            # NOTE: use worker to calculate
            self._print(f"Calculate reference frames {prefix} with potential...")
            cached_pred_fpath = self.directory / prefix / "pred.xyz"
            if not cached_pred_fpath.exists():
                worker.directory = self.directory / prefix
                worker.batchsize = nframes

                worker._share_wdir = True

                worker.run(ref_frames)
                worker.inspect(resubmit=True)
                if worker.get_number_of_running_jobs() == 0:
                    ret = worker.retrieve(
                        include_retrieved=True,
                    )
                    pred_frames = list(itertools.chain(*ret))
                else:
                    ...
                write(cached_pred_fpath, pred_frames)
            else:
                pred_frames = read(cached_pred_fpath, ":")
        else:
            ...
        pred_symbols, pred_energies, pred_forces = get_properties(pred_frames)

        # - get reaction coordinate (dimer distance here)
        a1, a2, a3 = self.angle[1], self.angle[0], self.angle[2]

        ref_angles = []
        for atoms in ref_frames:
            ang = atoms.get_angle(a1, a2, a3, mic=True)
            ref_angles.append(ang)

        # - save data
        abs_errors = [x - y for x, y in zip(pred_energies, ref_energies)]
        rel_errors = [(x / y) * 100.0 for x, y in zip(abs_errors, ref_energies)]
        data = np.array(
            [ref_angles, ref_energies, pred_energies, abs_errors, rel_errors]
        ).T

        np.savetxt(
            self.directory / prefix / f"{prefix}.dat",
            data,
            fmt="%8.4f  %12.4f  %12.4f  %12.4f  %8.4f",
            header="{:<8s}  {:<12s}  {:<12s}  {:<12s}  {:<8s}".format(
                "ang", "ref", "mlp", "abs", "rel [%]"
            ),
        )

        # - plot data
        fig, ax = plt.subplots(
            nrows=1, ncols=1, gridspec_kw={"hspace": 0.3}, figsize=(16, 9)
        )
        plt.suptitle(f"{prefix} with nframes {nframes}")

        ax.plot(
            ref_angles,
            ref_energies,
            marker="o",
            markerfacecolor="w",
            label="Reference",
        )
        ax.plot(
            ref_angles,
            pred_energies,
            marker="o",
            markerfacecolor="w",
            label="Prediction",
        )

        ax.set_ylabel("Energy [eV]")
        ax.set_xlabel("Dimer Distance [Å]")

        ax.legend()

        if (self.directory / f"{prefix}.png").exists():
            warnings.warn(f"Figure file {prefix} exists.", UserWarning)
        else:
            plt.savefig(self.directory / prefix / f"{prefix}.png")

        return


if __name__ == "__main__":
    ...
