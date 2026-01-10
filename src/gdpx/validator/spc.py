import copy
import itertools
import re
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

try:
    plt.style.use("presentation")
except Exception:
    ...

from gdpx.data.array import AtomsNDArray
from gdpx.worker.drive import DriverBasedWorker

from ..utils.comparision import get_properties, plot_distribution, plot_parity
from .validator import BaseValidator


class SinglepointValidator(BaseValidator):
    """Calculate energies on each structures and save them to file."""

    def __init__(
        self,
        subsets: Optional[list[str]] = None,
        groups: Optional[dict] = None,
        convergence: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """Init a spc validator.

        Args:
            groups: Errors are estimated for given groups.

        """
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.convergence = convergence

        self.subsets = subsets

        return

    def run(self, dataset, worker: Optional[DriverBasedWorker] = None) -> bool:
        """"""
        super().run()

        assert worker is not None, "SinglePointValidator must have a worker assigned."
        assert worker.driver.setting.task == "spc", "SinglePointValidator must use a driver with spc task."

        # Load previous rmse.dat
        if (self.directory / "rmse.dat").exists():
            ...
        else:
            ...

        for subname, subdata in dataset.items():
            if isinstance(subdata, Mapping):
                ...
            else:
                if isinstance(subdata, AtomsNDArray):
                    dataset[subname] = {("set",): subdata.get_marked_structures()}
                else:
                    raise Exception(f"Unknown input structures for {subname} -> {type(subdata)}.")

        get_data = lambda k: (
            dataset["reference"][k],
            dataset["prediction"].get(k) if "prediction" in dataset else None,
        )

        # Load structures
        is_spc_finished = True
        if self.subsets is None:
            data, frame_pairs = [], []
            for tuple_path in dataset["reference"].keys():
                prefix = "+".join(tuple_path)
                v_frames, p_frames = get_data(tuple_path)
                pred_frames = self._irun(prefix, v_frames, p_frames, worker)
                if pred_frames is None:
                    is_spc_finished = False
                    continue
                nframes, rmse_ret = self._plot_comparison(prefix, v_frames, pred_frames)
                frame_pairs.append([v_frames, pred_frames])
                data.append([prefix, nframes, rmse_ret])
        else:
            # TODO: use tree structure
            group_names = {k: [] for k in self.subsets}
            group_structures = {k: [] for k in self.subsets}
            for tuple_path, frames in dataset["reference"].items():
                for subset in self.subsets:
                    if tuple_path[0].startswith(f"{subset}"):
                        group_names[subset].append(tuple_path)
                        group_structures[subset].extend(frames)
            for k, v in group_names.items():
                with open(self.directory / f"subset-{k}.txt", "w") as fopen:
                    fopen.write("\n".join(["/".join(x) for x in v]))
            for k in self.subsets:
                num_systems = len(group_names[k])
                num_frames = len(group_structures[k])
                if num_systems > 0 and num_frames > 0:
                    ...
                else:
                    raise RuntimeError(f"No structures for subset {k}.")
            data = []
            for subset in self.subsets:
                frames = group_structures[subset]
                pred_frames = self._irun(subset, frames, None, worker)
                if pred_frames is None:
                    is_spc_finished = False
                    continue
                nframes, rmse_ret = self._plot_comparison(subset, frames, pred_frames)
                data.append([subset, nframes, rmse_ret])
        self.write_data(data)

        return is_spc_finished

    def write_data(self, data, fname: str = "rmse.dat"):
        """"""
        # - check data file
        keys = ["ene", "frc"]
        for rmse_ret in [x[2] for x in data]:
            for k in rmse_ret.keys():
                if k not in keys:
                    keys.append(k)
        content_fmt = "{:<48s}  {:>8d}  " + "{:>8.4f}  {:>8.4f}  " * len(keys) + "\n"

        header_fmt = "{:<48s}  {:>8s}  " + "{:>8s}  {:>8s}  " * len(keys) + "\n"
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

        with open(self.directory / fname, "w") as fopen:
            fopen.write(content)
        for l in content.split("\n"):
            self._print(l)

        return

    def _irun(
        self,
        prefix: str,
        ref_frames: list[Atoms],
        pred_frames: Optional[list[Atoms]],
        worker,
    ):
        """"""
        # - read structures
        num_frames = len(ref_frames)
        if pred_frames is None:
            self._print(f"calculate reference frames {prefix} with potential...")
            worker.directory = self.directory / prefix
            worker.batchsize = num_frames
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
        else:
            ...

        return pred_frames

    def _plot_comparison(self, prefix, ref_frames: list[Atoms], pred_frames: list[Atoms]):
        """"""
        if not (self.directory / prefix).exists():
            (self.directory / prefix).mkdir(parents=True)

        # FIXME: Check if reference structures have properties such as energies and forces
        nframes = len(ref_frames)
        ref_symbols, ref_energies, ref_forces = get_properties(ref_frames)
        ref_natoms = [len(a) for a in ref_frames]
        _, pred_energies, pred_forces = get_properties(pred_frames)

        # - figure
        fig, axarr = plt.subplots(nrows=1, ncols=2, gridspec_kw={"hspace": 0.3}, figsize=(16, 9))
        axarr = axarr.flatten()
        fig.suptitle(f"{prefix} with nframes {nframes}")

        # -- energies
        ene_rmse = plot_parity(axarr[0], ref_energies, pred_energies, x_name="ene", weights=ref_natoms)

        # -- forces
        frc_rmse = plot_parity(axarr[1], ref_forces, pred_forces, x_name="frc", x_types=ref_symbols)

        # if (self.directory/f"{prefix}.png").exists():
        #    warnings.warn(f"Figure file {prefix} exists.", UserWarning)
        fig.savefig(self.directory / prefix / "rmse.png", bbox_inches="tight")
        plt.close()

        # plot distributions
        fig, axarr = plt.subplots(nrows=1, ncols=2, gridspec_kw={"hspace": 0.3}, figsize=(16, 9))
        axarr = axarr.flatten()
        plt.suptitle(f"{prefix} with nframes {nframes}")

        plot_distribution(axarr[0], ref_energies, pred_energies, x_name="ene", weights=ref_natoms)
        plot_distribution(axarr[1], ref_forces, pred_forces, x_name="frc", x_types=ref_symbols)

        plt.savefig(self.directory / prefix / "dist.png")
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

    def report_convergence(self):
        """"""
        converged = True

        rmse_fpath = self.directory / "rmse.dat"
        if rmse_fpath.exists():
            with open(rmse_fpath, "r") as fopen:
                lines = fopen.readlines()
            col_names = lines[0].strip()[1:].split()[1:]
            row_names = [x.strip().split()[0] for x in lines[1:]]
            data = np.array([x.strip().split()[1:] for x in lines[1:]], dtype=np.float32)

            if self.convergence is not None:
                convergence = copy.deepcopy(self.convergence)
                pattern = convergence.pop("pattern", None)
                if pattern is not None:
                    matched_names = [x for x in row_names if re.match(pattern, x)]
                else:
                    matched_names = row_names

                assert all([x in col_names for x in convergence.keys()]), "Unavailable keys for convergence."

                converged = True
                for name in matched_names:
                    self._print(name)
                    iname = row_names.index(name)
                    for k, v in convergence.items():
                        ik = col_names.index(k)
                        self._print(f"{data[iname, ik]} <=? {v}")
                        if data[iname, ik] <= v:
                            converged = True
                        else:
                            converged = False
            else:
                # No convergenc criteria is provided, set converged False
                converged = False
        else:
            self._print("No rmse data is available and set convergence to True.")

        self._print(f"    >>> {converged}")

        return converged
