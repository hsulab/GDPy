import pathlib
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.formula import Formula
from joblib import Parallel, delayed

try:
    plt.style.use("presentation")  # type: ignore
except Exception as e:
    ...

from gdpx.geometry.align import wrap_traj
from gdpx.group import evaluate_group_expression

from ..data.array import AtomsNDArray
from .validator import BaseValidator


def plot_msd(
    wdir,
    names,
    lagtimes,
    timeseries,
    start_step=20,
    end_step=60,
    prefix="",
    print_func=print,
):
    """"""
    # - get self-diffusivity
    from scipy.stats import linregress

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
    fig.suptitle("MSD")

    # ax.set_title("$51Å\times 44Å$")
    if names is None:
        names = [str(i) for i in range(len(lagtimes))]

    for name, x, y in zip(names, lagtimes, timeseries):
        if y.shape[0] == 1:
            time, msd_avg = x, y[0]
            # show mean_squared_displacement
            (p1,) = ax.plot(time, msd_avg, label=f"{name}")

            # compute diffusion coefficient
            if not (start_step < 0 or start_step >= end_step):
                linear_model = linregress(msd_avg[start_step:end_step], time[start_step:end_step])
                ax.plot(
                    msd_avg[[start_step, end_step]],
                    time[[start_step, end_step]],
                    marker="o",
                    markerfacecolor="w",
                    color=p1.get_color(),
                )

                slope = linear_model.slope
                error = linear_model.rvalue
                # dim_fac is 3 as we computed a 3D msd with 'xyz'
                D = slope * 1 / (2 * 3)
                ax.text(np.median(time), np.median(msd_avg), f"$D={D:>.2e}$")
                print_func(f"system {name} diffusion_coefficient: {D} [Ang^2/ps]")
            else:
                ...
        elif y.shape[0] == 2:
            # show mean_squared_displacement
            time = x
            msd_avg, msd_std = y
            (p1,) = ax.plot(time, msd_avg, label=f"{name}")
            intv = time.shape[0] // 10
            ax.errorbar(
                time[::intv],
                msd_avg[::intv],
                yerr=msd_std[::intv],
                linestyle="None",
                marker="o",
                markerfacecolor="white",
                color=p1.get_color(),
            )
        else:
            raise Exception(f"Invalid shape {y.shape}.")

    ax.set_ylabel(r"MSD [Å$^2$]")
    ax.set_xlabel("Time [ps]")

    ax.legend(fontsize=12)

    fig.savefig(wdir / f"{prefix}msd.png", bbox_inches="tight")

    return


def compute_mean_squared_displacement(
    frames: list[Atoms],
    lagmax: int,
    start: int,
    end: int,
    timeintv: float,
    get_group_positions: Callable,
):
    """Compute mean squared displacement (MSD) for a group of atoms.

    Args:
        frames: List of ASE Atoms objects representing the trajectory frames.
        lagmax: Maximum lag time for MSD calculation.
        start: Start index for the trajectory frames.
        end: End index for the trajectory frames.
        timeintv: Time interval between frames in femtoseconds.
        get_group_positions: Function to extract group positions from Atoms object.

    Returns:
        lagtimes: Array of lag times in picoseconds.
        timeseries: Array of mean squared displacements for each lag time.

    """
    # Wrap the trajectory to avoid jump across periodic boundaries.
    frames = wrap_traj(frames)
    frames = frames[start:end:]

    positions = []
    for atoms in frames:
        positions.append(get_group_positions(atoms))
    positions = np.array(positions)

    _, natoms, _ = positions.shape

    # Compute the mean squared displacement (MSD) for each lag time.
    msds_by_particle = np.zeros((lagmax, natoms))

    lagtimes = np.arange(1, lagmax)
    for lag in lagtimes:
        disp = positions[:-lag, :, :] - positions[lag:, :, :]
        sqdist = np.square(disp).sum(axis=-1)
        msds_by_particle[lag, :] = np.mean(sqdist, axis=0)
    timeseries = msds_by_particle.mean(axis=1)

    timeintv = timeintv / 1000.0  # fs to ps
    lagtimes = np.arange(lagmax) * timeintv

    return lagtimes, timeseries


def compute_mean_squared_displacement_over_trajectories(
    frames_list: list[list[Atoms]],
    lagmax: int,
    start: Optional[int],
    end: Optional[int],
    timeintv: float,
    get_group_positions: Callable,
):
    """Compute mean squared displacement (MSD) for a group of atoms.

    Args:
        frames_list: List of lists of ASE Atoms objects representing the trajectory frames.
        lagmax: Maximum lag time for MSD calculation.
        start: Start index for the trajectory frames.
        end: End index for the trajectory frames.
        timeintv: Time interval between frames in femtoseconds.
        get_group_positions: Function to extract group positions from Atoms object.

    Returns:
        lagtimes: Array of lag times in picoseconds.
        timeseries: Array of mean squared displacements for each lag time.

    """
    # Wrap the trajectory to avoid jump across periodic boundaries.
    positions_list = []
    for frames in frames_list:
        frames = wrap_traj(frames)
        frames = frames[start:end:]

        positions = []
        for atoms in frames:
            positions.append(get_group_positions(atoms))
        positions = np.array(positions)  # (num_frames, num_atoms, 3)

        positions_list.append(positions)

    _, _, _ = positions_list[0].shape

    # Compute the mean squared displacement (MSD) for each lag time.
    msd_avg = np.zeros((lagmax, 1))
    msd_std = np.zeros((lagmax, 1))

    lagtimes = np.arange(1, lagmax)
    for lag in lagtimes:
        disp2_list = []
        for positions in positions_list:
            disp = positions[:-lag, :, :] - positions[lag:, :, :]
            sqdist = np.square(disp).sum(axis=-1)  # (nframes-lag, num_atoms)
            disp2 = np.mean(sqdist, axis=1)  # (nframes-lag,)
            disp2_list.append(disp2)
        disp2 = np.concatenate(disp2_list, axis=0)  # (nframes-lag,)
        msd_avg[lag] = np.mean(disp2, axis=0)
        msd_std[lag] = np.std(disp2, axis=0)

    msd_avg = msd_avg.flatten()
    msd_std = msd_std.flatten()

    timeintv = timeintv / 1000.0  # fs to ps
    lagtimes = np.arange(lagmax) * timeintv

    return lagtimes, (msd_avg, msd_std)


class MeanSquaredDisplacementValidator(BaseValidator):
    """Estimate the diffusion coefficient.

    Compute a windowed MSD where the MSD is averaged over all possible lag-times
    tau < tau_max.

    """

    def __init__(
        self,
        timeintv: float,
        lagmax: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        d_start: int = -1,
        d_end: int = 20,
        merge_trajs: bool = False,
        group: Optional[str] = None,
        com_group: Optional[str] = None,
        directory: Union[str, pathlib.Path] = "./",
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(directory, *args, **kwargs)

        self.group = group
        self.com_group = com_group

        self.start = start
        self.end = end

        self.lagmax = lagmax
        self.timeintv = timeintv

        # - diffusion coefficient linear fitting
        self.d_start = d_start
        self.d_end = d_end

        # Other parameters
        self.merge_trajs = merge_trajs

        return

    def _process_data(self, data) -> list[list[Atoms]]:
        """"""
        data = AtomsNDArray(data)
        self._debug(f"data: {data}")

        if data.ndim == 1:
            data = [data.tolist()]
        elif data.ndim == 2:  # assume it is from minimisations...
            data = data.tolist()
        else:
            raise RuntimeError(f"Invalid shape {data.shape}.")

        return data

    def run(self, dataset: dict, worker=None, *args, **kwargs) -> bool:
        """"""
        super().run()

        is_finished = True

        # Find some optional parameters
        labels = kwargs.get("labels", None)

        # Process reference and prediction data
        self._print("process reference ->")
        reference = dataset.get("reference")
        if reference is not None:
            self._irun(reference, "ref-", labels)

        self._print("process prediction ->")
        prediction = dataset.get("prediction")
        if prediction is not None:
            self._irun(prediction, "pre-", labels)

        return is_finished

    def _irun(self, data, prefix="", labels=None):
        """Test the first trajectory.

        TODO: Several trajectories.

        """
        mdtrajs = self._process_data(data)

        # TODO: Check trajectory consistency in number of atoms, elements, pbc...
        chemical_symbols = mdtrajs[0][0].get_chemical_symbols()
        if self.group is not None:
            group_indices = evaluate_group_expression(mdtrajs[0][0], self.group)
            group_formula = Formula.from_list([chemical_symbols[i] for i in group_indices]).convert("metal")
            self._print(f"num_atoms in the group: {len(group_indices)} formula: {group_formula}")
            self._debug(f"group_indices: {group_indices}")
        else:
            group_indices = None

        if self.com_group is not None:
            com_group_indices = evaluate_group_expression(mdtrajs[0][0], self.com_group)
            com_group_formula = Formula.from_list([chemical_symbols[i] for i in com_group_indices]).convert("metal")
            self._print(f"num_atoms in the com_group: {len(com_group_indices)} formula: {com_group_formula}")
            self._debug(f"com_group_indices: {com_group_indices}")
            # Check index in com_group should be a subset of group_indices
            if group_indices is not None:
                if not set(com_group_indices).issubset(set(group_indices)):
                    raise RuntimeError("com_group should be a subset of group.")
        else:
            com_group_indices = None

        cache_msd = self.directory / f"{prefix}msd.npy"
        if not cache_msd.exists():
            # Check if we use center of mass
            if com_group_indices is not None:
                get_group_positions = lambda atoms: atoms.get_positions()[com_group_indices, :].mean(axis=0)[
                    np.newaxis, :
                ]
            else:
                if group_indices is not None:
                    get_group_positions = lambda atoms: atoms.get_positions()[group_indices, :]
                else:
                    get_group_positions = lambda atoms: atoms.get_positions()

            if not self.merge_trajs:
                raw_data = Parallel(n_jobs=self.njobs)(
                    delayed(compute_mean_squared_displacement)(
                        [a for a in frames if a is not None],  # AtomsNDArray may have None...
                        lagmax=self.lagmax,
                        start=self.start,
                        end=self.end,
                        timeintv=self.timeintv,
                        get_group_positions=get_group_positions,
                    )
                    for frames in mdtrajs
                )
                data = np.array(raw_data)
            else:
                raw_data = compute_mean_squared_displacement_over_trajectories(
                    [[a for a in frames if a is not None] for frames in mdtrajs],
                    lagmax=self.lagmax,
                    start=self.start,
                    end=self.end,
                    timeintv=self.timeintv,
                    get_group_positions=get_group_positions,
                )
                data = np.vstack(raw_data)[np.newaxis, :]  # (1, 3, lagmax)
            np.save(cache_msd, data)
        else:
            data = np.load(cache_msd)

        lagtimes = [x[0] for x in data]
        timeseries = [x[1:] for x in data]
        plot_msd(
            self.directory,
            names=labels,
            lagtimes=lagtimes,
            timeseries=timeseries,
            start_step=self.d_start,
            end_step=self.d_end,
            prefix=prefix,
            print_func=self._print,
        )

        return
