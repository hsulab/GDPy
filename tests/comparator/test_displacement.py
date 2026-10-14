import copy

import numpy as np
import pytest
from ase import Atoms

from gdpx.analysis.comparators import create_comparator
from gdpx.analysis.comparators.displacement import DisplacementComparator
from gdpx.exploration.population.comparators import create_population_comparator


def distance(first, second, **parameters):
    comparator = DisplacementComparator(random_seed=0, **parameters)
    return comparator.fingerprint_distance(*comparator.prepare_data([first, second]))


def test_mic_is_applied_before_permutation_assignment():
    first = Atoms("Cu2", positions=[[0.1, 0, 0], [4.9, 0, 0]], cell=[10] * 3, pbc=[True, False, False])
    second = Atoms("Cu2", positions=[[5.1, 0, 0], [9.9, 0, 0]], cell=first.cell, pbc=first.pbc)
    assert distance(first, second, permutation_invariant=True, mic=True) == pytest.approx(0.2)
    assert distance(first, second, permutation_invariant=True, mic=False) == pytest.approx(5.0)
    assert distance(first, second, permutation_invariant=False, mic=True) == pytest.approx(5.0)


@pytest.mark.parametrize("permutation_invariant", [False, True])
def test_mic_respects_partial_periodicity(permutation_invariant):
    first = Atoms("Cu", positions=[[9.9, 0, 9.9]], cell=[10] * 3, pbc=[True, True, False])
    second = first.copy()
    second.positions[:] = [0.1, 0, 0.1]
    assert distance(first, second, permutation_invariant=permutation_invariant, mic=True) == pytest.approx(np.hypot(0.2, 9.8))


def test_mic_with_triclinic_cell():
    cell = np.array([[10, 0, 0], [4, 8, 0], [0, 0, 10]])
    first = Atoms("Cu", positions=[0.99 * cell[1]], cell=cell, pbc=True)
    second = Atoms("Cu", positions=[0.01 * cell[1]], cell=cell, pbc=True)
    assert distance(first, second, permutation_invariant=True, mic=True) == pytest.approx(0.02 * np.linalg.norm(cell[1]))


def test_matching_never_exchanges_different_elements():
    first = Atoms("CuPt", positions=[[0, 0, 0], [2, 0, 0]])
    second = first.copy()
    second.positions[:] = second.positions[::-1]
    assert distance(first, second, permutation_invariant=True) == pytest.approx(2)


def test_fixed_index_order_is_preserved():
    first = Atoms("CuPt", positions=[[0, 0, 0], [2, 0, 0]])
    second = first[[1, 0]]
    with pytest.raises(ValueError, match="matching species at each index"):
        distance(first, second)
    assert distance(first, second, permutation_invariant=True) == pytest.approx(0)
    first = Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]])
    second = first[[1, 0]]
    assert distance(first, second) == pytest.approx(2)
    assert distance(first, second, permutation_invariant=True) == pytest.approx(0)


def test_rigid_translation_and_rotation_remain_visible():
    first = Atoms("CuPtO", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0]])
    second = first.copy()
    second.translate([3, 4, 0])
    assert distance(first, second, permutation_invariant=True) == pytest.approx(5)
    second = first.copy()
    second.rotate(90, "z")
    assert distance(first, second, permutation_invariant=True) > 1


def test_group_selection_and_preparation_do_not_mutate_input():
    first = Atoms("MgCuPt", positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0]], cell=[10] * 3, pbc=True)
    first.info["marker"] = [1, 2]
    second = first.copy()
    second.positions[0] += [1, 2, 3]
    saved_positions = first.positions.copy()
    saved_info = copy.deepcopy(first.info)
    comparator = DisplacementComparator(group_expression="`id 2:3`", mic=True, random_seed=0)
    assert comparator(first, second)
    prepared = comparator.prepare_data([first, second])
    prepared[0].positions += 10
    np.testing.assert_array_equal(first.positions, saved_positions)
    assert first.info == saved_info


@pytest.mark.parametrize("alter", ["cell", "pbc", "zero_cell"])
def test_invalid_mic_geometry(alter):
    first = Atoms("Cu", positions=[[0, 0, 0]], cell=[10] * 3, pbc=True)
    second = first.copy()
    if alter == "cell":
        second.cell *= 2
    elif alter == "pbc":
        second.pbc = False
    else:
        first.cell = second.cell = [0, 0, 0]
    with pytest.raises(ValueError, match="MIC comparison requires"):
        distance(first, second, mic=True)


def test_threshold_is_rms_displacement_and_strict():
    first = Atoms("CuPt", positions=[[0, 0, 0], [2, 0, 0]])
    second = first.copy()
    second.positions[:, 0] += [0.3, 0.4]
    expected = np.sqrt((0.3 ** 2 + 0.4 ** 2) / 2)
    assert distance(first, second) == pytest.approx(expected)
    # Compare a stable single-atom boundary case exactly.
    first = Atoms("Cu", positions=[[0, 0, 0]])
    second = Atoms("Cu", positions=[[0.5, 0, 0]])
    comparator = DisplacementComparator(rmsd_max=0.5, random_seed=0)
    assert not comparator(first, second)


@pytest.mark.parametrize("parameters", [dict(rmsd_max=0), dict(rmsd_max=np.nan), dict(mic=1),
                                        dict(permutation_invariant="true"), dict(align=True)])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        DisplacementComparator(random_seed=0, **parameters)


def test_population_supplies_mic_and_no_presets_remain():
    for periodic in (False, True):
        comparator = create_population_comparator(dict(method="displacement", permutation_invariant=True), periodic=periodic)
        assert comparator.mic is periodic
    for method in ("cartesian_coordinate", "permutation_rmsd"):
        with pytest.raises(KeyError):
            create_comparator(dict(method=method))
