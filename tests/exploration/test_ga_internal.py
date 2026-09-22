import numpy as np
import pytest
from ase import Atoms

from gdpx.exploration.genetic_algorithm.core import RandomStreamRegistry
from gdpx.exploration.genetic_algorithm.crossover import (
    PeriodicCutAndSpliceCrossover,
)
from gdpx.exploration.genetic_algorithm.mutation.group_rattle import GroupRattleMutation
from gdpx.exploration.genetic_algorithm.mutation.rattle import RattleMutation
from gdpx.exploration.genetic_algorithm.operators import (
    CROSSOVERS,
    MUTATIONS,
    instantiate_a_genetic_operator,
)
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


def test_rattle_names_are_unambiguous():
    assert MUTATIONS["rattle"] is RattleMutation
    assert MUTATIONS["group_rattle"] is GroupRattleMutation
    assert "rattle_buffer" not in MUTATIONS


def test_crossover_names_are_explicit():
    assert CROSSOVERS == {"cut_and_splice": PeriodicCutAndSpliceCrossover}


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        ("periodic_cut_and_splice", "cut_and_splice"),
        ("cluster_cut_and_splice", "cut_and_splice"),
        ("cut_and_splice_cluster", "cut_and_splice"),
    ],
)
def test_renamed_crossovers_report_the_replacement(old_name, new_name):
    with pytest.raises(ValueError, match=new_name):
        instantiate_a_genetic_operator("crossover", {"method": old_name}, {})
