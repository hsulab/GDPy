import copy
import importlib

import numpy as np
import pytest
from ase import Atoms

from gdpx.structures.regions import HemisphereRegion
from gdpx.structures.regions.factory import create_region


@pytest.mark.parametrize("normal", [(0, 0, 1), (0, 0, -2), (1, 2, 3), (1e300, 0, 0), (0, 1e-300, 0)])
def test_uniform_sampling(normal):
    region = HemisphereRegion([3, -4, 2], 2, normal)
    positions = region.get_random_positions(6000, np.random.default_rng(42))
    offsets = positions - region._origin
    assert all(region._is_within_region(position) for position in positions)
    # A uniform half-ball has E[r^3] = R^3/2 and E[projection] = 3R/8.
    assert np.mean(np.linalg.norm(offsets, axis=1)**3) == pytest.approx(4, abs=0.15)
    assert np.mean(offsets @ region._normal) == pytest.approx(0.75, abs=0.04)
    np.testing.assert_array_equal(
        positions, region.get_random_positions(6000, np.random.default_rng(42))
    )


def test_boundaries_volume_and_contained_atoms():
    region = HemisphereRegion([1, 2, 3], 2)
    positions = [[1, 2, 3], [1, 2, 5], [3, 2, 3], [1, 2, 2], [1, 2, 5.01]]
    atoms = Atoms("H5", positions=positions)
    assert region.get_contained_indices(atoms) == [0, 1, 2]
    assert region.get_volume() == pytest.approx(16 * np.pi / 3)
    # An oblique normal also includes its flat face and rejects the opposite side.
    oblique = HemisphereRegion([1, 2, 3], 2, [1, 1, 0])
    assert oblique._is_within_region([2, 1, 3])
    assert oblique._is_within_region([2, 3, 3])
    assert not oblique._is_within_region([0, 1, 3])


def test_factory_and_serialization():
    config = {"method": "hemisphere", "origin": [1, 2, 3], "radius": 2, "normal": [1, 2, 3]}
    original = copy.deepcopy(config)
    region = create_region(config)
    assert config == original
    assert isinstance(region, HemisphereRegion)
    assert create_region(region) is region
    restored = create_region(region.as_dict())
    np.testing.assert_allclose(restored._normal, region._normal)
    np.testing.assert_allclose(restored._origin, region._origin)
    assert restored.get_volume() == region.get_volume()


@pytest.mark.parametrize("suffix, normal", [("", [0, 0, 1]), (" 0 0 -2", [0, 0, -1])])
def test_from_str(suffix, normal):
    region = HemisphereRegion.from_str("hemisphere 1 2 3 2" + suffix)
    np.testing.assert_allclose(region._origin, [1, 2, 3])
    np.testing.assert_allclose(region._normal, normal)
    assert region._radius == 2


@pytest.mark.parametrize("overrides", [
    {"radius": 0}, {"radius": -1}, {"radius": np.inf}, {"radius": np.nan},
    {"normal": [0, 0, 0]}, {"normal": [1, 2]}, {"normal": [0, np.inf, 1]},
    {"normal": [0, np.nan, 1]}, {"origin": [0, 1]}, {"origin": [0, 0, np.nan]},
])
def test_invalid_geometry(overrides):
    config = {"origin": [0, 0, 0], "radius": 1}
    config.update(overrides)
    with pytest.raises(ValueError):
        HemisphereRegion(**config)


@pytest.mark.parametrize("command", ["hemisphere", "hemisphere 0 0 0", "hemisphere 0 0 0 1 0"])
def test_invalid_command(command):
    with pytest.raises(ValueError):
        HemisphereRegion.from_str(command)


@pytest.mark.parametrize("module, name", [
    ("base", "BaseRegion"), ("auto", "AutoRegion"), ("cube", "CubeRegion"),
    ("sphere", "SphereRegion"), ("cylinder", "CylinderRegion"), ("lattice", "LatticeRegion"),
    ("surface", "SurfaceRegion"),
    ("intersect", "IntersectRegion"), ("hemisphere", "HemisphereRegion"),
])
def test_compatibility_imports(module, name):
    legacy = importlib.import_module("gdpx.structures.regions.region")
    new = importlib.import_module("gdpx.structures.regions." + module)
    assert getattr(legacy, name) is getattr(new, name)


@pytest.mark.parametrize("config", [
    {"method": "auto"},
    {"method": "cube", "origin": [0, 0, 0], "boundary": [0, 0, 0, 1, 1, 1]},
    {"method": "sphere", "origin": [0, 0, 0], "radius": 2},
    {"method": "cylinder", "origin": [0, 0, 0], "radius": 2, "height": 3},
    {"method": "lattice", "origin": [0, 0, 0], "cell": np.eye(3).tolist()},
])
def test_existing_region_factory(config):
    region = create_region(config)
    restored = create_region(region.as_dict())
    assert type(restored) is type(region)
    assert restored.as_dict() == region.as_dict()
    region.preprocess(Atoms("H", cell=np.eye(3), pbc=True))
    assert region.get_volume() > 0
    assert region.get_random_positions(1, np.random.default_rng(42)).shape == (1, 3)
    assert region._is_within_region(np.array([0.0, 0.0, 0.0]))
