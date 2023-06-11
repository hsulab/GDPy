#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import warnings
import pathlib
from typing import List, Union

import numpy as np
import numpy.ma as ma

from ase import Atoms
from ase.io import read, write

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    print("Used default matplotlib style.")

from GDPy.validator.validator import AbstractValidator
from GDPy.worker.drive import DriverBasedWorker

from GDPy.computation.utils import make_clean_atoms
from GDPy.utils.comparasion import parity_plot_dict, rms_dict

from GDPy.core.register import registers
from GDPy.validator.utils import get_properties
from GDPy.utils.command import convert_indices


def plot_distribution(ax, x_ref, x_pred, x_name="data", x_types=None, weights=None):
    """"""
    x_ref, x_pred = np.array(x_ref), np.array(x_pred)
    assert x_ref.shape[0] == x_pred.shape[0], "Input data is inconsistent."

    if weights is None:
        weights = np.ones(x_ref.shape)
    weights = np.array(weights)
    assert x_ref.shape[0] == weights.shape[0], "Weight is inconsistent."

    # use flat shape otherwise hist will be separate
    x_diff = (x_pred - x_ref) / weights
    #print("x_diff: ", x_diff.shape)
    #if len(x_diff.shape) > 1:
    #    x_diff = np.linalg.norm(x_diff, axis=1)

    pmax, pmin = np.max(x_diff), np.min(x_diff)

    ax.set_ylabel("Probability Density")
    ax.set_xlabel("$\Delta$"+x_name)

    num_bins = 20
    if x_types is None:
        x_diff = x_diff.flatten()
        n, bins, patches = ax.hist(x_diff, num_bins, density=True)
    else:
        # -- per type
        x_types = np.array(x_types)
    
        types = sorted(set(x_types))
        for t in types:
            t_mask = np.array(x_types==t)
            x_diff_t = x_diff[t_mask]
            if len(x_diff.shape) > 1:
                x_diff_t = x_diff_t.flatten()
            n, bins, patches = ax.hist(x_diff_t, num_bins, density=True, label=t)

    ax.legend()

    return


def plot_parity(ax, x_ref, x_pred, x_name="data", x_types=None, weights=None):
    """Plots the distribution of energy per atom on the output vs the input."""
    # - convert data type
    x_ref, x_pred = np.array(x_ref), np.array(x_pred)
    assert x_ref.shape[0] == x_pred.shape[0], "Input data is inconsistent."

    if weights is None:
        weights = np.ones(x_ref.shape)
    weights = np.array(weights)
    assert x_ref.shape[0] == weights.shape[0], "Weight is inconsistent."

    x_ref /= weights
    x_pred /= weights

    # - get the appropriate limits for the plot
    pmax = np.max(np.array([x_ref,x_pred])) # property max
    pmin = np.min(np.array([x_ref,x_pred]))
    edge = (pmax-pmin)*0.05

    plim = (pmin - edge, pmax + edge)
    ax.set_xlim(plim)
    ax.set_ylim(plim)

    # add line of slope 1 for refrence
    ax.plot(plim, plim, c="k")

    # - set labels
    ax.set_title(x_name)

    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")

    # - either plot points all togather or per type
    x_rmse = [rms_dict(x_ref, x_pred)]
    x_rmse_names = [x_name]

    if x_types is None:
        # -- all togather
        ax.scatter(x_ref, x_pred, label=x_name)
        # -- rmse text
        add_rmse_text(ax, x_rmse, x_rmse_names)
    else:
        # -- per type
        x_types = np.array(x_types)

        types = sorted(set(x_types))
        for t in types:
            t_mask = np.array(x_types==t)
            #print(t_mask)
            x_ref_t, x_pred_t = x_ref[t_mask], x_pred[t_mask]
            #print(t, x_ref_t.shape)
            ax.scatter(x_ref_t, x_pred_t, label=t)

            _rms = rms_dict(x_ref_t, x_pred_t)
            x_rmse.append(_rms)
            x_rmse_names.append(t)
        
        # -- rmse text
        add_rmse_text(ax, x_rmse, x_rmse_names)
    
    ax.legend()

    return x_rmse, x_rmse_names

def add_rmse_text(ax, x_rmse, x_name):
    """"""
    # add text about RMSE
    rmse_text = "RMSE:\n"
    for _rms, name in zip(x_rmse, x_name):
        rmse_text += "{:>6.3f}+-{:>6.3f} {:<4s}\n".format(_rms["rmse"], _rms["std"], name)

    ax.text(
        0.9, 0.1, rmse_text, transform=ax.transAxes, 
        fontsize=18, fontweight="bold", 
        horizontalalignment="right", verticalalignment="bottom"
    )

    return

@registers.validator.register
class SinglepointValidator(AbstractValidator):

    """Calculate energies on each structures and save them to file.
    """

    def run(self, dataset, worker: DriverBasedWorker, *args, **kwargs):
        """"""
        super().run()

        data = []
        frame_pairs = []
        for prefix, frames in dataset:
            pred_frames = self._irun(prefix, frames, None, worker)
            nframes, rmse_ret = self._plot_comparison(prefix, frames, pred_frames)
            frame_pairs.append([frames, pred_frames])
            data.append([prefix, nframes, rmse_ret])
        self.write_data(data)

        # - plot specific groups
        task_params = copy.deepcopy(self.task_params)
        def run_selection():
            #prefixes = [d[0] for d in dataset]
            #print(prefixes)

            selected_prefixes, selected_groups = [], []
            for k, v in task_params.items():
                selected_prefixes.append(k)
                selected_groups.append(
                    convert_indices(v, index_convention="py")
                )
            print(selected_groups)
            print(selected_prefixes)

            for curr_prefix, curr_indices in zip(selected_prefixes,selected_groups):
                curr_ref = list(itertools.chain(*[frame_pairs[i][0] for i in curr_indices]))
                curr_pre = list(itertools.chain(*[frame_pairs[i][1] for i in curr_indices]))
                nframes, rmse_ret = self._plot_comparison(curr_prefix, curr_ref, curr_pre)
                self.write_data([[curr_prefix, nframes, rmse_ret]], f"{curr_prefix}-rmse.dat")
        
        if task_params is not None:
            run_selection()

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
        self.logger.info(content)

        return

    def _irun(self, prefix: str, ref_frames: List[Atoms], pred_frames: List[Atoms], worker):
        """"""
        # - read structures
        nframes = len(ref_frames)
        if pred_frames is None:
            # NOTE: use worker to calculate
            # TODO: use cached data?
            self.logger.info(f"Calculate reference frames {prefix} with potential...")
            cached_pred_fpath = self.directory / prefix / "pred.xyz"
            if not cached_pred_fpath.exists():
                worker.directory = self.directory / prefix
                worker.batchsize = nframes

                worker._share_wdir = True

                worker.run(ref_frames)
                worker.inspect(resubmit=True)
                if worker.get_number_of_running_jobs() == 0:
                    pred_frames = worker.retrieve(
                        ignore_retrieved=False,
                    )
                else:
                    # TODO: ...
                    ...
                write(cached_pred_fpath, pred_frames)
            else:
                pred_frames = read(cached_pred_fpath, ":")
        
        return pred_frames
    
    def _plot_comparison(self, prefix, ref_frames, pred_frames: List[Atoms]):
        """"""
        if not (self.directory/prefix).exists():
            (self.directory/prefix).mkdir(parents=True)

        nframes = len(ref_frames)
        ref_symbols, ref_energies, ref_forces = get_properties(ref_frames)
        ref_natoms = [len(a) for a in ref_frames]
        pred_symbols, pred_energies, pred_forces = get_properties(pred_frames)
        
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