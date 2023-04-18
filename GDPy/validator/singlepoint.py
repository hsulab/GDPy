#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
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
plt.style.use("presentation")

from GDPy.validator.validator import AbstractValidator

from GDPy.computation.utils import make_clean_atoms
from GDPy.utils.comparasion import parity_plot_dict, rms_dict

from GDPy.core.register import registers

def get_properties(frames: List[Atoms], other_props = []):
    """Get properties of frames for comparison.

    Currently, only total energy and forces are considered.

    Returns:
        tot_symbols: shape (nframes,)
        tot_energies: shape (nframes,)
        tot_forces: shape (nframes,3)

    """
    tot_symbols, tot_energies, tot_forces = [], [], []

    for atoms in frames: # free energy per atom
        # -- basic info
        symbols = atoms.get_chemical_symbols()
        tot_symbols.extend(symbols)

        # -- energy
        energy = atoms.get_potential_energy() 
        tot_energies.append(energy)

        # -- force
        forces = atoms.get_forces(apply_constraint=False)
        tot_forces.extend(forces.tolist())

    return tot_symbols, tot_energies, tot_forces

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

    def run(self):
        """"""
        super().run()
        params = self.task_params

        # - check structures
        ref_fpaths = params["structures"].get("reference", None)
        if ref_fpaths is None:
            raise RuntimeError("No reference structures were found.")
        
        pred_fpaths = params["structures"].get("prediction", None)
        if pred_fpaths is not None:
            assert len(ref_fpaths) == len(pred_fpaths), "Inconsistent reference and prediction structures."
        else:
            pred_fpaths = [None]*len(ref_fpaths)
        
        prefix_names = params["structures"].get("names", None)
        if prefix_names is not None:
            assert len(ref_fpaths) == len(prefix_names), "Inconsistent names of structures."
        else:
            prefix_names = [None]*len(ref_fpaths)
        
        # - run calculations...
        data = []
        for prefix, ref_fpath, pred_fpath in zip(prefix_names, ref_fpaths, pred_fpaths):
            if prefix is None:
                prefix = pathlib.Path(ref_fpath).stem
            nframes, rmse_ret = self._irun(prefix, ref_fpath, pred_fpath)
            #print(rmse_ret)
            data.append([prefix, nframes, rmse_ret])

        # - check data file
        keys = ["en", "frc"]
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

        with open(self.directory/"rmse.dat", "w") as fopen:
            fopen.write(content)
        self.logger.info(content)

        return
    
    def _irun(self, prefix: str, ref: Union[str,pathlib.Path], pred: Union[str,pathlib.Path]=None):
        """"""
        # - read structures
        ref_frames = read(ref, ":")
        ref_symbols, ref_energies, ref_forces = get_properties(ref_frames)

        nframes = len(ref_frames)
        ref_natoms = [len(a) for a in ref_frames]

        if pred is None:
            # NOTE: use worker to calculate
            # TODO: use cached data?
            self.logger.info(f"Calculate reference frames {prefix} with potential...")
            cached_pred_fpath = self.directory / prefix / "pred.xyz"
            if not cached_pred_fpath.exists():
                cur_worker = self.worker
                cur_worker.directory = self.directory / prefix
                cur_worker.batchsize = nframes

                cur_worker._compact = True

                cur_worker.run(ref_frames, use_wdir=True)
                cur_worker.inspect(resubmit=True)
                if cur_worker.get_number_of_running_jobs() == 0:
                    pred_frames = self.worker.retrieve(
                        ignore_retrieved=False,
                    )
                else:
                    # TODO: ...
                    ...
                write(cached_pred_fpath, pred_frames)
            else:
                pred_frames = read(cached_pred_fpath, ":")
        else:
            pred_frames = read(pred, ":")
        pred_symbols, pred_energies, pred_forces = get_properties(pred_frames)
        
        # - figure
        fig, axarr = plt.subplots(
            nrows=1, ncols=2,
            gridspec_kw={"hspace": 0.3}, figsize=(16, 9)
        )
        axarr = axarr.flatten()

        plt.suptitle(f"{prefix} with nframes {nframes}")

        # -- energies
        en_rmse = plot_parity(
            axarr[0], ref_energies, pred_energies, x_name="en", weights=ref_natoms
        )

        # -- forces
        frc_rmse = plot_parity(
            axarr[1], ref_forces, pred_forces, x_name="frc", x_types=ref_symbols
        )

        if (self.directory/f"{prefix}.png").exists():
            warnings.warn(f"Figure file {prefix} exists.", UserWarning)

        plt.savefig(self.directory/f"{prefix}.png")

        # - save results to data file
        #with open(self.directory/"rmse.dat", "a") as fopen:
        #    content = "{:<24s}  {:>8d}  ".format(prefix, nframes)
        #    content += "{:>8.4f}  {:>8.4f}  ".format(
        #        en_rmse[0][0]["rmse"], en_rmse[0][0]["std"]
        #    )
        #    fopen.write(content)
        rmse_ret = {}
        x_rmse, x_rmse_names = en_rmse
        for _rms, rms_name in zip(x_rmse, x_rmse_names):
            rmse_ret[rms_name] = _rms
        x_rmse, x_rmse_names = frc_rmse
        for _rms, rms_name in zip(x_rmse, x_rmse_names):
            rmse_ret[rms_name] = _rms

        return nframes, rmse_ret


if __name__ == "__main__":
    ...