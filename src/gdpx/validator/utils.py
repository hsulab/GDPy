import numpy as np
import numpy.typing
from ase import Atoms
from scipy.interpolate import make_interp_spline


def get_properties(frames: list[Atoms], other_props=[]):
    """Get properties of frames for comparison.

    Currently, only total energy and forces are considered.

    Returns:
        tot_symbols: shape (nframes,)
        tot_energies: shape (nframes,)
        tot_forces: shape (nframes,3)

    """
    tot_symbols, tot_energies, tot_forces = [], [], []

    for atoms in frames:  # free energy per atom
        # -- basic info
        symbols = atoms.get_chemical_symbols()
        tot_symbols.extend(symbols)

        # -- energy
        energy = atoms.get_potential_energy()
        tot_energies.append(energy)

        # -- force
        forces = atoms.get_forces(apply_constraint=False)
        tot_forces.extend(forces.tolist())


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
