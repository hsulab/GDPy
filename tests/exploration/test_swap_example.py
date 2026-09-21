import copy
from pathlib import Path

import numpy as np
import yaml

from gdpx.exploration.factory import create_expedition
from gdpx.exploration.genetic_algorithm.mutation.swap import SwapMutation


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "global_optimisation"
    / "explorations/genetic_algorithm/cu7ni6.yaml"
)


def test_cu7ni6_example_generates_tagged_clusters_and_swaps_species():
    config = yaml.safe_load(EXAMPLE_PATH.read_text())
    config.pop("scheduler", None)
    engine = create_expedition(config)[0]

    candidates = engine.builders["random"].run(size=4)
    for atoms in candidates:
        assert atoms.get_chemical_formula() == "Cu7Ni6"
        assert np.array_equal(np.sort(atoms.get_tags()), np.arange(1, 14))

    parent = candidates[0]
    parent.info = {
        "confid": 1,
        "data": {},
        "key_value_pairs": {},
    }
    mutation_config = copy.deepcopy(engine.ga_dict["operators"]["mutation"])
    assert mutation_config.pop("method") == "swap"
    mutation = SwapMutation(
        bond_distance_dict=engine.generator.get_bond_distance_dict(),
        rng=np.random.default_rng(31),
        **mutation_config,
    )

    child, description = mutation.get_new_individual([parent])

    assert child is not None
    assert description == "mutation: swap [1/10]"
    assert child.get_chemical_formula() == parent.get_chemical_formula()
    assert np.array_equal(child.get_tags(), parent.get_tags())
    assert not np.allclose(child.positions, parent.positions)

    parent_symbols = np.array(parent.get_chemical_symbols())
    child_symbols = np.array(child.get_chemical_symbols())
    parent_cu_positions = parent.positions[parent_symbols == "Cu"]
    parent_ni_positions = parent.positions[parent_symbols == "Ni"]
    child_cu_positions = child.positions[child_symbols == "Cu"]
    child_ni_positions = child.positions[child_symbols == "Ni"]
    assert any(
        np.isclose(position, parent_ni_positions, atol=1e-12).all(axis=1).any()
        for position in child_cu_positions
    )
    assert any(
        np.isclose(position, parent_cu_positions, atol=1e-12).all(axis=1).any()
        for position in child_ni_positions
    )
