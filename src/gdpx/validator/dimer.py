#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("presentation")
except Exception as e:
    ...

from ase import Atoms
from ase.io import read, write

from gdpx.builder.builder import StructureBuilder
from gdpx.worker.drive import DriverBasedWorker

from .validator import BaseValidator


def compare_structures(v_frames: list[Atoms], p_frames: list[Atoms]):
    v_ene = np.array([a.get_potential_energy() for a in v_frames])
    p_ene = np.array([a.get_potential_energy() for a in p_frames])

    distances = np.array([a.get_distance(0, 1, mic=True) for a in v_frames])

    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])

    results = dict(
        distances=distances[sorted_indices],
        v_ene=v_ene[sorted_indices],
        p_ene=p_ene[sorted_indices],
    )

    return results


def summarise_validation(distances, v_ene, p_ene):
    """"""
    content = "# dis [Å]  ref [eV]  mlp [eV]  abs [eV]  rel [%]\n"
    for dis, v, p in zip(distances, v_ene, p_ene):
        abs_error = p - v
        rel_error = (abs_error / v) * 100.0
        content += f"{dis:8.4f}  {v:12.4f}  {p:12.4f}  {abs_error:12.4f}  {rel_error:8.4f}\n"

    return content


def plot_dimer_curve(fig_fpath: pathlib.Path, dimer: str, distances, v_ene, p_ene):
    """"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
    assert isinstance(ax, plt.Axes)

    ax.set_title(dimer)

    ax.plot(
        distances,
        v_ene,
        marker="o",
        markerfacecolor="w",
        label="Reference",
    )
    ax.plot(
        distances,
        p_ene,
        marker="o",
        markerfacecolor="w",
        label="Prediction",
    )

    ax.set_ylabel("Energy [eV]")
    ax.set_xlabel("Distance [Å]")

    ax.legend()

    fig.savefig(fig_fpath, bbox_inches="tight")

    return


class DimerValidator(BaseValidator):

    name: str = "dimer"

    def run(
        self,
        structures: Optional[Any] = None,
        worker: Optional[DriverBasedWorker] = None,
        *args,
        **kwargs,
    ) -> bool:
        """"""
        super().run()

        if worker is not None:
            v_worker = worker
            self._print("Use the worker at run time.")
        else:
            v_worker = self.worker
        assert (
            v_worker is not None
        ), "Worker must be provided either init or run time."
        v_worker.directory = self.directory / "_run"

        if structures is not None:
            v_structures = structures
            self._print("Use the structures at run time.")
        else:
            v_structures = self.structures
        self._print(f"{v_structures=}")
        assert (
            v_structures is not None
        ), "Structures must be provided either init or run time."

        if isinstance(v_structures, StructureBuilder):
            v_structures = v_structures.run()

        # Make sure all structures are dimers
        num_atoms = [len(a) for a in v_structures]
        if any([n != 2 for n in num_atoms]):
            raise Exception("All structures must be dimers.")
        composition = [a.get_chemical_formula() for a in v_structures]
        if any([c != composition[0] for c in composition]):
            raise Exception("All dimers must be the same.")

        # Run calculations
        is_finished = False

        p_structures = self._irun(v_structures, v_worker)
        if p_structures is not None:
            results = compare_structures(v_structures, p_structures)
            if not pathlib.Path(self.directory / "v.dat").exists():
                content = summarise_validation(**results)
                with open(self.directory / "v.dat", "w") as fopen:
                    fopen.write(content)
            fig_fpath = self.directory / "dimer.png"
            plot_dimer_curve(fig_fpath, composition[0], **results)
        else:
            is_finished = False

        return is_finished

    def _irun(
        self, frames: list[Atoms], worker: DriverBasedWorker
    ) -> Optional[list[Atoms]]:
        """"""
        assert isinstance(
            worker, DriverBasedWorker
        ), "Worker must be a DriverBasedWorker."

        cache_fpath = self.directory / "pred.xyz"
        if cache_fpath.exists():
            end_frames = read(cache_fpath, ":")
            return end_frames  # type: ignore

        _ = worker.run(frames)
        _ = worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            trajectories = worker.retrieve(include_retrieved=True)
            end_frames = [t[-1] for t in trajectories]
            write(cache_fpath, end_frames)  # type: ignore
        else:
            end_frames = None

        return end_frames  # type: ignore


if __name__ == "__main__":
    ...
