"""Tag preservation for atom-wise mirror mutation."""

import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.genetic_algorithm.mutation.mirror import MirrorMutation
from gdpx.structures.geometry.ga import closest_distances_generator


@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("n_top", [4, 5])
def test_mirror_preserves_tags_at_parent_atom_indices(reflect, n_top):
    parent = Atoms(
        symbols=["Cu", "Cu", "Ni", "Cu", "Ni"],
        positions=[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 2.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 2.0, 0.0],
        ],
        tags=[0, 7, 3, 7, 11] if n_top == 4 else [2, 7, 3, 7, 11],
        cell=[15.0, 15.0, 15.0],
        pbc=False,
    )
    parent.info["confid"] = 12
    original = parent.copy()
    mutation = MirrorMutation(
        blmin=closest_distances_generator(parent.numbers, 0.5),
        n_top=n_top,
        reflect=reflect,
        use_tags=False,
        rng=np.random.default_rng(42),
    )

    child, description = mutation.get_new_individual([parent])

    assert child is not None
    assert child.has("tags")
    np.testing.assert_array_equal(child.get_tags(), original.get_tags())
    np.testing.assert_array_equal(child.numbers, original.numbers)
    np.testing.assert_array_equal(child.positions[:-n_top], original.positions[:-n_top])
    assert not np.array_equal(child.positions[-n_top:], original.positions[-n_top:])
    assert parent == original
    assert child.info["data"]["parents"] == [12]
    assert description == "mutation: mirror"
