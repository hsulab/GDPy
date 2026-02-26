import pathlib
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.formula import Formula
from ase.io import read, write
from joblib import Parallel, delayed
from scipy.stats import linregress

try:
    plt.style.use("presentation")  # type: ignore
except Exception as e:
    ...

from gdpx.geometry.align import wrap_traj
from gdpx.group import evaluate_group_expression

from ..data.array import AtomsNDArray
from .validator import BaseValidator


def string_to_index(stridx: str) -> tuple[int, int, int]:
    """Convert index string to a tuple of (start, end, step)."""
    if ":" in stridx:
        parts = [None if s == "" else int(s) for s in stridx.split(":")]
        num_parts = len(parts)
        if num_parts == 2:
            start, end = parts
            step = 1
        elif num_parts == 3:
            start, end, step = parts
        else:
            raise Exception(f"Invalid index string: {stridx}")
        if start is None:
            start = 0
        if end is None:
            raise Exception(f"Invalid index string: {stridx}, end cannot be None.")
        if step is None:
            step = 1
    else:
        start = int(stridx)
        end = start + 1
        step = 1

    return start, end, step


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
    # get self-diffusivity
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
            intv = 1 if intv < 1 else intv
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

    # ax.legend(fontsize=12)

    fig.savefig(wdir / f"{prefix}msd.png", bbox_inches="tight")

    return


def preprocess_a_single_trajectory(
    frames: list[Atoms],
    start: Optional[int],
    end: Optional[int],
    intv: Optional[int],
    block_size: Optional[int],
    get_group_positions: Callable,
    dump_file: Optional[pathlib.Path] = None,
) -> list[np.ndarray]:
    """"""
    # Wrap the trajectory to avoid jump across periodic boundaries.
    if dump_file is not None:
        if dump_file.exists():
            frames = read(dump_file, index=":")
        else:
            frames = wrap_traj(frames)
            write(dump_file, frames)
    else:
        frames = wrap_traj(frames)
    frames = frames[start:end:intv]

    positions = []
    for atoms in frames:
        positions.append(get_group_positions(atoms))
    positions = np.array(positions)  # (num_frames, num_particles, 3)

    positions_list = []
    if block_size is not None:
        # divide into blocks and discard the last block if it is not longer than lagmax
        num_frames = positions.shape[0]
        num_blocks = num_frames // block_size
        for iblock in range(num_blocks):
            block_positions = positions[iblock * block_size : (iblock + 1) * block_size, :, :]
            positions_list.append(block_positions)
    else:
        positions_list.append(positions)

    return positions_list


def compute_mean_squared_displacement_without_average(positions: np.ndarray, lag: int):
    """"""
    beg_indices = np.array([0], dtype=int)
    end_indices = beg_indices + lag
    disp = positions[beg_indices, :, :] - positions[end_indices, :, :]
    sqdist = np.square(disp).sum(axis=-1)
    disp2 = np.mean(sqdist, axis=1)  # average over particles, (num_windows,)

    return disp2


def compute_mean_squared_displacement_by_window_average(positions: np.ndarray, lag: int, lagspace: int):
    """"""
    beg_indices = np.arange(0, positions.shape[0] - lag, lagspace)
    end_indices = beg_indices + lag
    disp = positions[beg_indices, :, :] - positions[end_indices, :, :]  # (num_windows, num_particles, 3)
    sqdist = np.square(disp).sum(axis=-1)  # (num_windows, num_particles)
    disp2 = np.mean(sqdist, axis=1)  # average over particles, (num_windows,)

    return disp2


def compute_mean_squared_displacement(
    frames: list[Atoms],
    window_slice: Optional[tuple[int, int, int]],
    block_size: Optional[int],
    start: int,
    end: int,
    intv: Optional[int],
    timeintv: float,
    get_group_positions: Callable,
    dump_file: Optional[pathlib.Path] = None,
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
    if intv is None:
        intv = 1

    if dump_file is not None:
        dump_file.parent.mkdir(parents=True, exist_ok=True)

    positions_list = preprocess_a_single_trajectory(
        frames=frames,
        start=start,
        end=end,
        intv=intv,
        block_size=block_size,
        get_group_positions=get_group_positions,
        dump_file=dump_file,
    )

    if window_slice is not None:
        # Get lag from window
        lagmin, lagmax, lagspace = window_slice

        get_disp2 = lambda positions, lag: compute_mean_squared_displacement_by_window_average(
            positions, lag=lag, lagspace=lagspace
        )
    else:
        # infer lagmax from the length of trajectories
        lagmin = 1
        lagmax = min([positions.shape[0] for positions in positions_list])
        lagspace = 1

        get_disp2 = lambda positions, lag: compute_mean_squared_displacement_without_average(positions, lag=lag)

    # Compute the mean squared displacement (MSD) for each lag time.
    num_trajs = len(positions_list)
    msd_avg = np.zeros((num_trajs, lagmax, 1))  # average over particles

    lagtimes = np.arange(lagmin, lagmax)
    for lag in lagtimes:
        for itraj, positions in enumerate(positions_list):  # over trajectories/blocks
            disp2 = get_disp2(positions, lag=lag)
            msd_avg[itraj, lag] = np.mean(disp2, axis=0)  # average over windows

    msd_avg = msd_avg.squeeze(axis=-1)  # (num_trajs, lagmax)

    timeintv = intv * timeintv / 1000.0  # fs to ps
    lagtimes = np.arange(lagmax) * timeintv
    lagtimes = lagtimes[np.newaxis, :].repeat(num_trajs, axis=0)  # (num_trajs, lagmax)

    # stack results as (num_trajs, 2, lagmax)
    data = np.stack([lagtimes, msd_avg], axis=1)

    return data


def compute_mean_squared_displacement_over_blocks(
    frames_list: list[list[Atoms]],
    window_slice: Optional[tuple[int, int, int]],
    block_size: Optional[int],
    start: Optional[int],
    end: Optional[int],
    intv: Optional[int],
    timeintv: float,
    get_group_positions: Callable,
    dump_directory: Optional[pathlib.Path] = None,
):
    """"""
    if intv is None:
        intv = 1

    if dump_directory is not None:
        dump_directory.mkdir(parents=True, exist_ok=True)

    positions_list = []
    for itraj, frames in enumerate(frames_list):
        curr_positions_list = preprocess_a_single_trajectory(
            frames=frames,
            start=start,
            end=end,
            intv=intv,
            block_size=block_size,
            get_group_positions=get_group_positions,
            dump_file=dump_directory / f"traj-{itraj:>02d}.xyz" if dump_directory is not None else None,
        )
        positions_list.extend(curr_positions_list)

    if window_slice is not None:
        # Get lag from window
        lagmin, lagmax, lagspace = window_slice

        get_disp2 = lambda positions, lag: compute_mean_squared_displacement_by_window_average(
            positions, lag=lag, lagspace=lagspace
        )
    else:
        # infer lagmax from the length of trajectories
        lagmin = 1
        lagmax = min([positions.shape[0] for positions in positions_list])
        lagspace = 1

        get_disp2 = lambda positions, lag: compute_mean_squared_displacement_without_average(positions, lag=lag)

    # Compute the mean squared displacement (MSD) for each lag time.
    msd_avg = np.zeros((lagmax, 1))
    msd_std = np.zeros((lagmax, 1))

    lagtimes = np.arange(lagmin, lagmax)
    for lag in lagtimes:
        disp2_list = []
        for positions in positions_list:  # over trajectories/blocks
            disp2 = get_disp2(positions, lag=lag)
            disp2_list.append(disp2)
        disp2 = np.concatenate(disp2_list, axis=0)  # (all_windows,)
        msd_avg[lag] = np.mean(disp2, axis=0)
        msd_std[lag] = np.std(disp2, axis=0)

    msd_avg = msd_avg.flatten()
    msd_std = msd_std.flatten()

    timeintv = intv * timeintv / 1000.0  # fs to ps
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
        window: Optional[str] = None,
        block_size: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        intv: Optional[int] = None,
        d_start: int = -1,
        d_end: int = 20,
        merge_trajs: bool = False,
        group: Optional[str] = None,
        com_group: Optional[str] = None,
        save_wrapped: bool = False,
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
        self.intv = intv

        self.timeintv = timeintv

        # Parse window
        if window is not None:
            self.window_slice = string_to_index(window)
            # make sure we have positive end, and step should be smaller than end-start
            if self.window_slice[1] is not None and self.window_slice[1] <= 0:
                raise Exception(f"Invalid window end {self.window_slice[1]}.")
            if self.window_slice[2] >= self.window_slice[1] - self.window_slice[0]:
                raise Exception(
                    f"Invalid window step {self.window_slice[2]} for window size {self.window_slice[1] - self.window_slice[0]}."
                )
        else:
            self.window_slice = None

        # Parse block
        self.block_size = block_size

        # - diffusion coefficient linear fitting
        self.d_start = d_start
        self.d_end = d_end

        # Other parameters
        self.merge_trajs = merge_trajs

        self.save_wrapped = save_wrapped

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
                self._print("compute MSD for each trajectory by window average ->")
                raw_data = Parallel(n_jobs=self.njobs)(
                    delayed(compute_mean_squared_displacement)(
                        [a for a in frames if a is not None],  # AtomsNDArray may have None...
                        window_slice=self.window_slice,
                        block_size=self.block_size,
                        start=self.start,
                        end=self.end,
                        intv=self.intv,
                        timeintv=self.timeintv,
                        get_group_positions=get_group_positions,
                        dump_file=self.directory / "wrapped" / f"traj-{itraj:>02d}.xyz" if self.save_wrapped else None,
                    )
                    for itraj, frames in enumerate(mdtrajs)
                )
                # find the maximum length and cut all data to the same length
                min_length = min([x.shape[2] for x in raw_data])
                raw_data = [x[:, :, :min_length] for x in raw_data]
                data = np.concatenate(raw_data, axis=0)
                self._print(f"{data.shape=}")
            else:
                clean_trajs = [[a for a in frames if a is not None] for frames in mdtrajs]
                if self.block_size is None:
                    self._print("compute MSD for trajectories together by window average ->")
                else:
                    self._print("compute MSD for trajectories in blocks by window average ->")
                raw_data = compute_mean_squared_displacement_over_blocks(
                    clean_trajs,
                    window_slice=self.window_slice,
                    block_size=self.block_size,
                    start=self.start,
                    end=self.end,
                    intv=self.intv,
                    timeintv=self.timeintv,
                    get_group_positions=get_group_positions,
                    dump_directory=self.directory / "wrapped" if self.save_wrapped else None,
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
