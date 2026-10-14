import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.analysis.selectors import create_selector
from gdpx.data.array import AtomsNDArray
from gdpx.analysis.comparators.rmsd import permutation_rmsd


def candidate(distance, energy, force=0.01, reverse=False):
    atoms = Atoms("CuPtO", positions=[[0, 0, 0], [distance, 0, 0], [0, 2, 0]], cell=[20] * 3)
    if reverse:
        atoms = atoms[[2, 1, 0]]
        atoms.translate([3, 4, 5])
        atoms.rotate(37, "z")
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.full((3, 3), force))
    return atoms


@pytest.mark.parametrize("method", ["oganov_fingerprint", "permutation_rmsd"])
def test_energy_rank_convergence_and_permutation(tmp_path, method):
    high = candidate(2, -1)
    different = candidate(4, -2)
    lowest = candidate(2, -3, reverse=True)
    unconverged = candidate(6, -4, force=0.04)  # vector norm > .05
    data = AtomsNDArray([[high, different], [lowest, unconverged]])
    selector = create_selector(dict(method="minima", number=2, comparator=dict(method=method, **({"pbc": False} if method == "oganov_fingerprint" else {})), directory=tmp_path, random_seed=0))
    result = selector.select(data)
    assert [a.get_potential_energy() for a in result] == [-2, -3]
    assert np.array_equal(np.argwhere(data.markers), [[0, 1], [1, 0]])
    assert all("fingerprints" not in atoms.info for atoms in [high, different, lowest, unconverged])


def test_support_excluded_from_fingerprints(tmp_path):
    frames = []
    for i in range(2):
        atoms = Atoms("Mg", positions=[[i * 5, 0, 0]], cell=[20] * 3) + candidate(2, -i)
        atoms.calc = SinglePointCalculator(atoms, energy=-i, forces=np.zeros((4, 3)))
        frames.append(atoms)
    selector = create_selector(dict(method="minima", number=2, comparator=dict(method="oganov_fingerprint", pbc=False, group_expression="`symbol Cu Pt O`"), directory=tmp_path, random_seed=0))
    with pytest.raises(ValueError, match="found only 1"):
        selector.select(frames)
    assert not selector.info_fpath.exists()


@pytest.mark.parametrize("parameters", [{"number": 0}, {"number": 1.5}, {"fmax": 0}, {"n_top": -1}])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        create_selector(dict(method="minima", random_seed=0, **parameters))


def test_nonfinite_properties(tmp_path):
    selector = create_selector(dict(method="minima", number=1, comparator=dict(method="permutation_rmsd"), directory=tmp_path, random_seed=0))
    with pytest.raises(ValueError, match="finite energies and forces"):
        selector.select([candidate(2, np.nan)])


def test_rmsd_merges_small_distortions(tmp_path):
    original = candidate(2, -3)
    distorted = candidate(2.2, -2, reverse=True)
    different = candidate(4, -1)
    assert permutation_rmsd(original, candidate(2, -3, reverse=True)) < 1e-10
    assert permutation_rmsd(original, distorted) < 0.5
    assert permutation_rmsd(original, different) > 0.5
    selector = create_selector(dict(method="minima", number=2, comparator=dict(method="permutation_rmsd", rmsd_max=0.5), directory=tmp_path, random_seed=0))
    assert [a.get_potential_energy() for a in selector.select([original, distorted, different])] == [-3, -1]


@pytest.mark.parametrize("parameter", ["n_top", "ofp", "rmsd"])
def test_obsolete_fields_have_migration_guidance(parameter):
    with pytest.raises(ValueError, match="group_expression"):
        create_selector(dict(method="minima", random_seed=0, **{parameter: 1}))


def test_comparator_is_explicit_and_requires_preparation_interface():
    with pytest.raises(ValueError, match="explicit comparator"):
        create_selector(dict(method="minima", random_seed=0))
    with pytest.raises(TypeError, match="prepare_data and compare_fingerprints"):
        create_selector(dict(method="minima", random_seed=0,
                             comparator=dict(method="cartesian_coordinate", random_seed=0)))


def test_force_checks_still_include_support(tmp_path):
    atoms = Atoms("Mg", positions=[[10, 0, 0]], cell=[20] * 3) + candidate(2, -1)
    forces = np.zeros((4, 3))
    forces[0, 0] = 0.1
    atoms.calc = SinglePointCalculator(atoms, energy=-1, forces=forces)
    selector = create_selector(dict(method="minima", number=1, directory=tmp_path, random_seed=0,
                                   comparator=dict(method="permutation_rmsd", group_expression="`id 2:4`")))
    with pytest.raises(ValueError, match="0 force-converged"):
        selector.select([atoms])
