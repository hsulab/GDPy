import numpy as np
import pytest
from ase import Atoms

from gdpx.structures.groups.group import evaluate_group_expression


def slab(z, periodic=True):
    return Atoms('H' * len(z), scaled_positions=[[0, 0, x] for x in z],
                 cell=[[4, 0, 0], [1, 3, 0], [0.5, 0, 20]], pbc=[True, True, periodic])


@pytest.mark.parametrize('bounds', ['', ' -0.1 0.9', ' 2.9 3.9', ' -3.1 -2.1'])
def test_shifted_intervals_and_multiple_images(bounds):
    atoms = slab([0.95, 0.25, 0.75, -5.05, 7.25])
    original = atoms.positions.copy()
    assert set(evaluate_group_expression(atoms, f'`zbot 2{bounds}`')) == {0, 3}
    assert evaluate_group_expression(atoms, f'`ztop 1{bounds}`') == [2]
    np.testing.assert_array_equal(atoms.positions, original)


def test_default_seam_and_custom_interval():
    atoms = slab([-0.1, 0.9, 3.9, -4.1, 0.5])
    assert set(evaluate_group_expression(atoms, '`zbot 4`')) == {0, 1, 2, 3}
    assert evaluate_group_expression(atoms, '`ztop 1`') == [4]
    assert evaluate_group_expression(atoms, '`zbot 1 0 1`') == [4]


def test_nonperiodic_axis_and_zero_count():
    atoms = slab([-2, 0.5, 3], periodic=False)
    assert evaluate_group_expression(atoms, '`zbot 1 2 3`') == [0]
    assert evaluate_group_expression(atoms, '`ztop 1`') == [2]
    assert evaluate_group_expression(atoms, '`ztop 0`') == []


@pytest.mark.parametrize('arguments', ['-1', '2 0', '2 0 1 extra', '2 0 2',
                                        '2 1 0', '2 nan 1', '2 0 inf'])
@pytest.mark.parametrize('method', ['zbot', 'ztop'])
def test_invalid_position_groups(method, arguments):
    with pytest.raises(ValueError):
        evaluate_group_expression(slab([0.1]), f'`{method} {arguments}`')
