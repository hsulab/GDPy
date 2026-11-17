import numpy as np
import pytest
from ase import Atoms
from gdpx.data.array import AtomsNDArray
from gdpx.workflow.factory import create_selector


def trajectory(xs):
    return [Atoms('H', positions=[[x, x, x]], cell=[10, 10, 10]) for x in xs]


def select(tmp_path, frames, **kwargs):
    data = AtomsNDArray(frames)
    selector = create_selector(dict(method='cartesian_rmse', rmse_interval=0.1, **kwargs))
    selector.directory = tmp_path
    selector.select(data)
    return np.argwhere(data.markers).tolist()


def test_interval_is_from_last_selected_not_path_length(tmp_path):
    assert select(tmp_path, trajectory([0, .06, 0, .06, .12, .15])) == [[4], [5]]


def test_endpoints_and_single_frame(tmp_path):
    assert select(tmp_path/'a', trajectory([0, .02]), include_first=True, include_last=False) == [[0]]
    assert select(tmp_path/'b', trajectory([0])) == [[0]]
    assert select(tmp_path/'c', trajectory([0]), include_last=False) == []
    # include_last is a force-inclusion switch, not an exclusion filter.
    assert select(tmp_path/'d', trajectory([0, .2]), include_last=False) == [[1]]


def test_cartesian_components_not_vector_rms(tmp_path):
    frames = [Atoms('H', positions=[[x, 0, 0]]) for x in [0, .15, .18]]
    assert select(tmp_path, frames, include_last=False) == [[2]]


def test_periodic_and_atom_group(tmp_path):
    frames = [Atoms('HO', positions=[[x, 0, 0], [0, 0, 0]], cell=[10]*3, pbc=True) for x in [9.95, .05, .3]]
    assert select(tmp_path/'a', frames, group_expression='`symbol H`') == [[2]]
    assert select(tmp_path/'b', frames, mic=False, group_expression='`symbol H`') == [[1], [2]]


def test_ragged_and_higher_dimensional_trajectories(tmp_path):
    data = [[trajectory([0, .2, .23]), trajectory([10, 10.02])], [trajectory([20, 20.01]), []]]
    assert select(tmp_path, data) == [[0, 0, 1], [0, 0, 2], [0, 1, 1], [1, 0, 1]]


def test_flat_trajectory_ids(tmp_path):
    frames = trajectory([0, .2, 20, 20.01])
    for a, name in zip(frames, ['a', 'a', 'b', 'b']): a.info['trajectory_id'] = name
    assert select(tmp_path, frames) == [[1], [3]]


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf'), True])
def test_bad_intervals(interval):
    with pytest.raises(ValueError):
        create_selector(dict(method='cartesian_rmse', rmse_interval=interval))


def test_invalid_trajectory(tmp_path):
    with pytest.raises(ValueError, match='ordering'):
        select(tmp_path/'a', [Atoms('H'), Atoms('O')])
    with pytest.raises(ValueError, match='empty'):
        select(tmp_path/'b', trajectory([0, .2]), group_expression='`symbol Ti`')
    frames = trajectory([0, .2]); frames[0].info['trajectory_id'] = 'a'
    with pytest.raises(ValueError, match='every frame'):
        select(tmp_path/'c', frames)
