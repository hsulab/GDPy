import numpy as np
from ase import Atoms

from gdpx.exploration.genetic_algorithm.core import RandomStreamRegistry
from gdpx.exploration.genetic_algorithm.mutation.standard import RattleMutation
from gdpx.structures.geometry.ga import atoms_too_close, closest_distances_generator


def test_named_random_streams_are_order_independent_and_restorable():
    first = RandomStreamRegistry(17)
    expected_a = first.get("a").random(4)
    expected_b = first.get("b").random(4)

    second = RandomStreamRegistry(17)
    actual_b = second.get("b").random(4)
    actual_a = second.get("a").random(4)
    np.testing.assert_allclose(actual_a, expected_a)
    np.testing.assert_allclose(actual_b, expected_b)

    state = first.snapshot()
    expected_next = first.get("a").random(4)
    restored = RandomStreamRegistry(17)
    restored.restore(state)
    np.testing.assert_allclose(restored.get("a").random(4), expected_next)


def test_rattle_uses_generator_and_preserves_valid_geometry():
    parent = Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    parent.info = {"confid": 1, "data": {}, "key_value_pairs": {}}
    minimum_distances = closest_distances_generator([29], 0.7)
    mutation = RattleMutation(
        minimum_distances,
        n_top=2,
        rattle_strength=0.2,
        rattle_prop=1.0,
        rng=np.random.default_rng(4),
    )

    child, description = mutation.get_new_individual([parent])

    assert child is not None
    assert description == "mutation: rattle"
    assert not atoms_too_close(child, minimum_distances)
    assert not np.allclose(child.positions, parent.positions)
