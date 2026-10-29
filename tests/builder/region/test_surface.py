import numpy as np
import pytest
from ase import Atoms

from gdpx.structures.regions import SurfaceRegion
from gdpx.structures.regions.factory import create_region
from gdpx.exploration.persist.thanos import dispatch_thanos
from gdpx.structures.builders.random_structure import RandomStructureImprovedModifier


def test_membership_is_periodic_xy_but_not_z():
    region = create_region(dict(method="surface", z_range=[2, 4]))
    atoms = Atoms("H6", positions=[[-100, 200, 2], [200, -100, 4],
                                  [0, 0, 1.9], [0, 0, 4.1], [0, 0, 22], [0, 0, -18]])
    assert region.get_contained_indices(atoms) == [0, 1]
    assert SurfaceRegion.from_str("surface 2 4").as_dict() == region.as_dict()


def test_sampling_volume_and_cell_refresh():
    cell = [[4, 0, 0], [1, 3, 0], [0, 0, 20]]
    region = SurfaceRegion([2, 4])
    with pytest.raises(RuntimeError):
        region.get_volume()
    region.preprocess(Atoms(cell=cell))
    positions = region.get_random_positions(2000, np.random.default_rng(5))
    uv = positions[:, :2] @ np.linalg.inv(np.asarray(cell)[:2, :2])
    assert np.all((uv >= 0) & (uv < 1))
    assert all(region._is_within_region(p) for p in positions)
    assert positions[:, 2].mean() == pytest.approx(3, abs=0.05)
    assert region.get_volume() == pytest.approx(24)
    assert region.as_dict() == dict(method="surface", z_range=[2, 4], wrap=[-0.1, 0.9])
    region.preprocess(Atoms(cell=[2, 2, 20]))
    assert region.get_volume() == pytest.approx(8)
    configured = SurfaceRegion([2, 4], cell=cell)
    configured.preprocess(Atoms(cell=[2, 2, 20]))
    assert create_region(configured.as_dict()).get_volume() == pytest.approx(24)


@pytest.mark.parametrize("wrap, periodic_z, expected", [(True, True, 1), (False, True, 0), (True, False, 0)])
def test_thanos_wraps_before_cartesian_check(wrap, periodic_z, expected):
    atoms = Atoms("O", positions=[[-12, 25, 23]], cell=[10, 10, 20], pbc=[True, True, periodic_z])
    original = atoms.positions.copy()
    rule = dispatch_thanos(name="region", group="`symbol O`",
                          region=dict(method="surface", z_range=[2, 4], wrap=wrap))
    assert rule(atoms) == expected
    np.testing.assert_array_equal(atoms.positions, original)


@pytest.mark.parametrize("bounds", [[2, 2], [4, 2], [0], [0, np.inf], [np.nan, 3]])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        SurfaceRegion(bounds)


@pytest.mark.parametrize("cell", [np.zeros((3, 3)), np.eye(2), [[4, 0, 1], [0, 4, 0], [0, 0, 20]]])
def test_invalid_sampling_cell(cell):
    with pytest.raises(ValueError):
        SurfaceRegion([2, 4], cell=cell)


def test_random_builder_uses_substrate_cell(tmp_path):
    substrate = Atoms("Cu", positions=[[0, 0, 0]], cell=[8, 8, 20], pbc=True)
    builder = RandomStructureImprovedModifier(
        substrates=[substrate], composition={"O": 1},
        region=dict(method="surface", z_range=[5, 7]), random_seed=12,
        directory=tmp_path,
    )
    frames = builder.run(size=1)
    assert len(frames) == 1
    oxygen = frames[0].positions[frames[0].numbers == 8]
    assert np.all((oxygen[:, 2] >= 5) & (oxygen[:, 2] <= 7))
    np.testing.assert_allclose(frames[0].cell, substrate.cell)


@pytest.mark.parametrize("wrap, expected", [(True, [0, 1]), ([0, 1], []), (False, [1])])
def test_direct_membership_and_tagged_particles_share_wrapping(wrap, expected):
    atoms = Atoms("O2", positions=[[0, 0, 19.5], [0, 0, -0.5]],
                  tags=[1, 2], cell=[10, 10, 20], pbc=True)
    # With default wrapping both atoms are at -0.5; [0, 1) moves both to 19.5.
    region = SurfaceRegion([-1, 0], wrap=wrap)
    original = atoms.positions.copy()
    assert region.get_contained_indices(atoms) == expected
    tags = region.get_contained_tags_dict(atoms).get("O", [])
    assert tags == [i + 1 for i in expected]
    np.testing.assert_array_equal(atoms.positions, original)
    assert create_region(region.as_dict()).get_contained_indices(atoms) == expected


def test_group_expression_surface_wrapping():
    from gdpx.structures.groups.group import evaluate_group_expression
    atoms = Atoms("O", positions=[[0, 0, 19.5]], cell=[10, 10, 20], pbc=True)
    assert evaluate_group_expression(atoms, "`region surface -1 0`") == [0]
    assert evaluate_group_expression(atoms, "`region surface -1 0 0 1`") == []
