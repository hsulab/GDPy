import copy
import itertools
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
from scipy.interpolate import make_interp_spline

try:
    plt.style.use("presentation")  # type: ignore
except Exception as e:
    ...

from ase import Atoms
from ase.neighborlist import neighbor_list
from joblib import Parallel, delayed

from gdpx.data.array import AtomsNDArray

from .validator import BaseValidator


def smooth_curve(
    bins: numpy.typing.NDArray,
    points: numpy.typing.NDArray,
    bspline_degree: int = 3,
    smooth_bin_interval: float = 0.01,
) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]:
    """Smooth the curve using B-spline interpolation.

    Args:
        bins: The bin centers.
        points: The values at the bin centers.
        bspline_degree: Degree of the B-spline.
        smooth_bin_interval: Interval for the smoothed bins.

    Returns:
        A tuple of smoothed bins and corresponding points.

    """
    # Get the spline representation of the data
    spl = make_interp_spline(bins, points, k=bspline_degree)

    # Create new bins with the specified interval
    new_bins = np.arange(bins.min(), bins.max() + smooth_bin_interval, smooth_bin_interval)

    # Evaluate the spline at the new bins
    new_points = spl(new_bins)
    new_points = np.where(new_points < 1e-6, 0, new_points)  # avoid very small values

    return new_bins, new_points


def compute_radial_distribution(
    working_directory: pathlib.Path,
    frames: list[Atoms],
    custom_pairs: list[str],
    volume: Optional[float] = None,
    nbins: int = 60,
    cutoff: float = 6.0,
    n_jobs: int = 1,
) -> dict:
    """Calculate radial distribution.

    Args:
        working_directory: Working directory that stores RDF results.
        frames: A List of Atoms objects.
        custom_pairs: Target species pairs, for example, ["Cu-Cu", "Cu-O"].
        volume: System volume.
        nbins: Number of bins for histogram.
        cutoff: Cut-off radius in Angstrom.
        n_jobs: Number of parallel jobs.

    """
    if not working_directory.exists():
        working_directory.mkdir(parents=True)

    # Parse RDF definition
    # We assume the system volume does not change along the trajectory!
    # if volume is None:
    #     volume = frames[0].get_volume()

    # The atom order should be consistent in the entire trajectory
    # i.e. this does not work for variable-composition system
    chemical_symbols = frames[0].get_chemical_symbols()
    species = list(set(chemical_symbols))
    sym_dict = {k: [] for k in species}
    for k, v in itertools.groupby(enumerate(chemical_symbols), key=lambda x: x[1]):
        sym_dict[k].extend([x[0] for x in v])
    all_pairs = ["-".join(p) for p in itertools.product(species, species)]

    pair_dict = {}
    for pair in custom_pairs:
        p0, p1 = pair.split("-")
        first_indices = copy.deepcopy(sym_dict.get(p0, []))
        second_indices = copy.deepcopy(sym_dict.get(p1, []))
        assert len(first_indices) > 0, f"Cant found {p0}."
        assert len(second_indices) > 0, f"Cant found {p1}."

        num_first, num_second = len(first_indices), len(second_indices)
        if p0 == p1:
            num_pairs = (num_first) * (num_second - 1)
        else:
            num_pairs = num_first * num_second
        pair_dict[pair] = num_pairs

    # Set up bins
    binwidth = cutoff / nbins
    bincentres = np.linspace(binwidth / 2.0, cutoff + binwidth / 2.0, nbins + 1)
    left_edges = np.copy(bincentres) - binwidth / 2.0
    _ = np.copy(bincentres) + binwidth / 2.0  # right_edges
    bins = np.linspace(0.0, cutoff + binwidth, nbins + 2)

    def compute_distance_histogram(atoms, all_pairs, custom_pairs, cutoff, bins, binwidth) -> dict:
        """"""
        i, j, d = neighbor_list("ijd", atoms, cutoff=cutoff + binwidth)

        symbols = atoms.get_chemical_symbols()
        num_pairs = len(d)

        distance_dict_ = {k: [] for k in all_pairs}
        for p in range(num_pairs):
            pair = f"{symbols[i[p]]}-{symbols[j[p]]}"
            distance_dict_[pair].append(d[p])

        distance_dict = {k: distance_dict_[k] for k in custom_pairs}

        dis_hist = {}
        for k, v in distance_dict.items():
            hist_, _ = np.histogram(v, bins)
            dis_hist[k] = hist_

        return dis_hist

    ret = Parallel(n_jobs=n_jobs)(
        delayed(compute_distance_histogram)(atoms, all_pairs, custom_pairs, cutoff, bins, binwidth) for atoms in frames
    )

    dis_hist = {k: [] for k in custom_pairs}
    for curr_dis_hist in ret:
        assert isinstance(curr_dis_hist, dict)
        for k, v in curr_dis_hist.items():
            dis_hist[k].append(v)

    # get pair density
    density_dict = {k: [] for k in custom_pairs}
    for atoms in frames:
        for k, num_pairs in pair_dict.items():
            if volume is None:
                density_dict[k].append(num_pairs / atoms.get_volume())
            else:
                density_dict[k].append(num_pairs / volume)

    # Reformat results
    results = {}
    for k, v in dis_hist.items():
        curr_dis_hist = np.array(v)
        avg_density = np.array(density_dict[k])[:, np.newaxis]

        # NOTE: VMD likely uses this formula
        vshells = 4.0 * np.pi * left_edges**2 * binwidth
        # vshells = 4./3.*np.pi*binwidth*(3*left_edges**2+3*left_edges*binwidth+binwidth**2)
        vshells[0] = 1.0  # avoid zero in division

        rdf = curr_dis_hist / vshells / avg_density

        rdf_avg = np.average(rdf, axis=0)
        rdf_min = np.min(rdf, axis=0)
        rdf_max = np.max(rdf, axis=0)
        rdf_svar = np.sqrt(np.var(rdf, axis=0))

        data = np.vstack((bincentres, rdf_avg, rdf_svar, rdf_min, rdf_max)).T
        np.savetxt(
            working_directory / f"{k}.dat",
            data,
            fmt="%8.4f  %8.4f  %8.4f  %8.4f  %8.4f",
            header=("{:<8s}  " * 5).format("r", "rdf", "svar", "min", "max"),
        )
        results[k] = data

    return results


def plot_radial_distribution_function(
    fig_path: pathlib.Path,
    prd_data: Optional[numpy.typing.NDArray] = None,
    ref_data: Optional[numpy.typing.NDArray] = None,
    title: str = "RDF",
    smooth_kwargs: Optional[dict] = None,
) -> None:
    """Plot radial distribution function.

    Args:
        fig_path: Path to save the figure.
        prd_data: Prediction data, shape (nbins, 2).
        ref_data: Reference data, shape (nbins, 2).
        title: Title of the plot.
        smooth_kwargs: Keyword arguments for smoothing.

    Returns:
        None.

    """
    fig = plt.figure(figsize=(12, 9))
    ax: plt.Axes = fig.subplots(1, 1)  # type: ignore

    ax.set_xlabel("r [Å]")
    ax.set_ylabel("g(r)")
    ax.set_title(title)

    if smooth_kwargs is None:
        smooth_kwargs = {}

    if prd_data is not None:
        bincentres, rdf = prd_data[:, 0], prd_data[:, 1]
        bincentres_, rdf_ = smooth_curve(bincentres, rdf, **smooth_kwargs)
        ax.plot(bincentres_, rdf_, ls="-", label="prediction")

    if ref_data is not None:
        bincentres, rdf = ref_data[:, 0], ref_data[:, 1]
        bincentres_, rdf_ = smooth_curve(bincentres, rdf, **smooth_kwargs)
        ax.plot(bincentres_, rdf_, ls="-.", label="reference")

    ax.legend()

    fig.savefig(fig_path, bbox_inches="tight")

    return


class RdfValidator(BaseValidator):
    def __init__(
        self,
        pairs: list[str],
        cutoff: float = 6.0,
        nbins: int = 60,
        smooth_kwargs: Optional[dict] = None,
        directory="./",
        *args,
        **kwargs,
    ) -> None:
        """Radial Distribution.

        Args:
            paris: A list of species pairs [Cu-Cu, ..., ...].
            cutoff: The radial cutoff radius in Angstrom.
            nbins: Number of bins.
            smooth_kwargs: Keyword arguments for smoothing the curve.

        """
        super().__init__(directory=directory, *args, **kwargs)

        self.pairs = pairs
        self.cutoff = cutoff
        self.nbins = nbins

        if smooth_kwargs is None:
            smooth_kwargs = {}
        self.smooth_kwargs = smooth_kwargs

        return

    def _process_data(self, data) -> list[Atoms]:
        """"""
        data = AtomsNDArray(data)

        if data.ndim == 1:
            data = [data.tolist()]
        elif data.ndim == 2:  # assume it is from extract_cache...
            data = data.tolist()
        elif data.ndim == 3:  # assume it is from a compute node...
            data_ = []
            for d in data[:]:  # TODO: add squeeze method?
                data_.extend(d)
            data = data_
        else:
            raise RuntimeError(f"Invalid shape {data.shape}.")

        return data[0]  # TODO: support several trajectories

    def run(self, dataset, worker=None, *args, **kwargs) -> bool:
        """Process reference and prediction data separately.

        TODO:

            Support average over several trajectories.

        """
        is_finished = True

        # Get custom volume, useful for surface with variable vacuum height
        volume = kwargs.get("volume", None)

        # Canonicalise input data format
        self._print("process reference ->")
        reference = dataset.get("reference", None)

        # Process reference and prediction data
        if reference is not None:
            ref_frames = self._process_data(reference)
            self._debug(f"reference  nframes: {len(ref_frames)}")
            ref_data = self._compute_radial_distribution(
                self.directory / "reference", ref_frames, self.pairs, self.cutoff, self.nbins, volume=volume
            )
        else:
            ref_data = None

        self._print("process prediction ->")
        prediction = dataset.get("prediction", None)
        if prediction is not None:
            pre_frames = self._process_data(prediction)
            self._debug(f"prediction nframes: {len(pre_frames)}")
            pre_data = self._compute_radial_distribution(
                self.directory / "prediction", pre_frames, self.pairs, self.cutoff, self.nbins, volume=volume
            )
        else:
            pre_data = None

        assert ref_data is not None or pre_data is not None, "Neither reference nor prediction is given."

        # compare results
        self._compare_results(ref_data, pre_data)

        return is_finished

    def _compute_radial_distribution(
        self,
        working_directory: pathlib.Path,
        frames: list[Atoms],
        pairs,
        cutoff,
        nbins,
        volume: Optional[float] = None,
    ):
        """"""
        if not working_directory.exists():
            data = compute_radial_distribution(
                working_directory, frames, pairs, volume, nbins, cutoff, n_jobs=self.njobs
            )
        else:
            data = {}
            saved_files = list(working_directory.glob("*.dat"))
            for p in saved_files:
                data[p.name[:-4]] = np.loadtxt(p)

        return data

    def _compare_results(self, reference, prediction):
        """"""
        for pair in self.pairs:
            p, r = None, None
            if prediction is not None:
                p = prediction.get(pair, None)
            if reference is not None:
                r = reference.get(pair, None)
            if not (p is None and r is None):
                plot_radial_distribution_function(
                    self.directory / f"{pair}_rdf.png", p, r, title=pair, smooth_kwargs=self.smooth_kwargs
                )
            else:
                if p is not None:
                    plot_radial_distribution_function(
                        self.directory / f"{pair}_rdf.png", p, None, title=pair, smooth_kwargs=self.smooth_kwargs
                    )
                else:  # if r is not None:
                    plot_radial_distribution_function(
                        self.directory / f"{pair}_rdf.png", None, r, title=pair, smooth_kwargs=self.smooth_kwargs
                    )

        return
