#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union, List

import numpy as np
import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    ...

from ase import Atoms
from ase.constraints import FixAtoms, constrained_indices

from ..core.node import AbstractNode
from ..utils.comparision import get_properties, plot_parity, plot_distribution
from ..builder.constraints import parse_constraint_info


def set_constraint(atoms, cons_text):
    """"""
    atoms._del_constraints()
    mobile_indices, frozen_indices = parse_constraint_info(
        atoms, cons_text, ignore_ase_constraints=True, ret_text=False
    )
    if frozen_indices:
        atoms.set_constraint(FixAtoms(indices=frozen_indices))

    return atoms


class SinglePointComparator(AbstractNode):

    def __init__(self, directory: Union[str, pathlib.Path] = "./", random_seed: int = None, *args, **kwargs):
        super().__init__(directory, random_seed, *args, **kwargs)

        return

    def run(self, prediction, reference):
        """"""
        # - input shape should be (?, ?, ?)
        print(f"reference: {reference}")
        print(f"prediction: {prediction}")

        reference = reference.get_marked_structures()
        prediction = prediction.get_marked_structures()

        #for a in reference:
        #    print(constrained_indices(a, only_include=FixAtoms))
        #    break

        assert len(reference) == len(prediction), "Number of structures are inconsistent."

        nframes, rmse_ret = self._plot_comparison("spc", reference, prediction)
        self.write_data([["spc", nframes, rmse_ret]])

        return

    def write_data(self, data, fname: str="rmse.dat"):
        """"""
        # - check data file
        keys = ["ene", "frc"]
        for rmse_ret in [x[2] for x in data]:
            for k in rmse_ret.keys():
                if k not in keys:
                    keys.append(k)
        content_fmt = "{:<24s}  {:>8d}  " + "{:>8.4f}  {:>8.4f}  "*len(keys) + "\n"

        header_fmt = "{:<24s}  {:>8s}  " + "{:>8s}  {:>8s}  "*len(keys) + "\n"
        header_data = ["#prefix", "nframes"]
        for k in keys:
            header_data.extend([f"{k}_rmse", f"{k}_std"])
        header = header_fmt.format(*header_data)
        
        content = header
        for prefix, nframes, rmse_ret in data:
            cur_data = [prefix, nframes]
            for k in keys:
                v = rmse_ret.get(k, None)
                if v is None:
                    cur_data.extend([np.nan, np.nan])
                else:
                    cur_data.extend([v["rmse"], v["std"]])
            content += content_fmt.format(*cur_data)
        
        with open(self.directory/fname, "w") as fopen:
            fopen.write(content)
        self._print(content)

        return

    def _plot_comparison(self, prefix, ref_frames: List[Atoms], pred_frames: List[Atoms]):
        """"""
        if not (self.directory/prefix).exists():
            (self.directory/prefix).mkdir(parents=True)

        nframes = len(ref_frames)
        ref_symbols, ref_energies, ref_forces = get_properties(ref_frames, apply_constraint=True)
        ref_natoms = [len(a) for a in ref_frames]
        pred_symbols, pred_energies, pred_forces = get_properties(pred_frames, apply_constraint=True)
        
        # - figure
        fig, axarr = plt.subplots(
            nrows=1, ncols=2,
            gridspec_kw={"hspace": 0.3}, figsize=(16, 9)
        )
        axarr = axarr.flatten()
        plt.suptitle(f"{prefix} with nframes {nframes}")

        # -- energies
        ene_rmse = plot_parity(
            axarr[0], ref_energies, pred_energies, x_name="ene", weights=ref_natoms
        )

        # -- forces
        frc_rmse = plot_parity(
            axarr[1], ref_forces, pred_forces, x_name="frc", x_types=ref_symbols
        )

        #if (self.directory/f"{prefix}.png").exists():
        #    warnings.warn(f"Figure file {prefix} exists.", UserWarning)
        plt.savefig(self.directory/prefix/"rmse.png")
        plt.close()

        # plot distributions
        fig, axarr = plt.subplots(
            nrows=1, ncols=2,
            gridspec_kw={"hspace": 0.3}, figsize=(16, 9)
        )
        axarr = axarr.flatten()
        plt.suptitle(f"{prefix} with nframes {nframes}")

        plot_distribution(
            axarr[0], ref_energies, pred_energies, x_name="ene", weights=ref_natoms
        )
        plot_distribution(
            axarr[1], ref_forces, pred_forces, x_name="frc", x_types=ref_symbols
        )

        plt.savefig(self.directory/prefix/"dist.png")
        plt.close()

        # - save results to data file
        rmse_ret = {}
        x_rmse, x_rmse_names = ene_rmse
        for _rms, rms_name in zip(x_rmse, x_rmse_names):
            rmse_ret[rms_name] = _rms
        x_rmse, x_rmse_names = frc_rmse
        for _rms, rms_name in zip(x_rmse, x_rmse_names):
            rmse_ret[rms_name] = _rms

        return nframes, rmse_ret


if __name__ == "__main__":
    ...