"""Public global-optimisation configuration and operator-defined parent selection."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from ase import Atoms

from gdpx.exploration.factory import create_exploration
from gdpx.exploration.population import PopulationBasedExploration
from gdpx.exploration.population.config import PopulationConfig
from gdpx.exploration.genetic_algorithm.selection import GeneticParentSelector
from gdpx.exploration.genetic_algorithm.generation import GeneticGenerationManager
from gdpx.exploration.population.population import compute_population_fitness

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/global_optimisation/explorations"


def config(method="genetic_algorithm"):
    return yaml.safe_load((EXAMPLES / method / "cu8.yaml").read_text())


def engine_for(parameters):
    result = create_exploration(parameters)
    return result[0] if isinstance(result, list) else result


def test_identical_population_can_be_used_by_both_strategies():
    ga = config()
    bh = config("basin_hopping")
    bh["population"] = copy.deepcopy(ga["population"])
    originals = copy.deepcopy((ga, bh))
    first, second = engine_for(ga), engine_for(bh)
    assert isinstance(first, PopulationBasedExploration)
    assert isinstance(second, PopulationBasedExploration)
    assert (ga, bh) == originals
    for engine in (first, second):
        engine.worker = SimpleNamespace(as_dict=lambda: {"schema_version": 3})
        saved = engine.as_dict()
        assert saved["method"] == "global_optimisation"
        assert "recipe" not in saved and "operators" not in saved
        assert saved["strategy"]["method"] in ("genetic_algorithm", "basin_hopping")
        saved.pop("runtime")
        assert type(engine_for(saved)) is type(engine)


@pytest.mark.parametrize("change,message", [
    (lambda c: c.pop("strategy"), "strategy mapping"),
    (lambda c: c["strategy"].pop("method"), "strategy.method"),
    (lambda c: c["strategy"].update(method=[]), "strategy.method"),
    (lambda c: c["strategy"].update(method="unknown"), "strategy.method"),
    (lambda c: c.update(recipe={}), "no recipe wrapper"),
    (lambda c: c.update(operators={}), "strategy.operators"),
    (lambda c: c["population"].update(name="variable"), "compatibility is automatic"),
    (lambda c: c["population"].update(substrate={}), "strategy.substrate"),
    (lambda c: c["population"]["generation"].update(mutation={}), "strategy.mutation"),
    (lambda c: c["strategy"].update(selection={"group_by_composition": False}), "automatically"),
    (lambda c: c["strategy"].update(num_mcmoves=1), "Unsupported"),
    (lambda c: c["strategy"].update(operators=[]), "must be a mapping"),
    (lambda c: c["strategy"]["reproduction"].update(size=3), "exceed"),
    (lambda c: c["strategy"].pop("completion"), "builder_proportions"),
])
def test_invalid_settings_fail_before_builder_construction(monkeypatch, change, message):
    parameters = config()
    change(parameters)
    monkeypatch.setattr(PopulationConfig, "initialise_builders",
                        lambda *args: pytest.fail("constructed builders for an invalid search"))
    with pytest.raises(ValueError, match=message):
        create_exploration(parameters)


def candidate(index, symbols="Cu2", tags=(1, 2), score=1.0):
    atoms = Atoms(symbols, positions=[[index, 0, 0], [index + 2, 0, 0]], tags=tags)
    atoms.info = {"confid": index, "key_value_pairs": {"raw_score": score}, "data": {}}
    return atoms


@pytest.mark.parametrize("allow_variable", [None, False, True])
@pytest.mark.parametrize("difference", ["composition", "ordering", "tags", "substrate", "empty_substrate", "none"])
def test_generation_uses_operator_compatibility(allow_variable, difference):
    first = candidate(1, "CuAg" if difference == "ordering" else "Cu2")
    second = candidate(2, "AgCu" if difference == "ordering" else
                       "Ag2" if difference == "composition" else "Cu2")
    if difference == "tags":
        second.set_tags([1, 1])
    if difference == "substrate":
        first.set_tags([0, 1])
        second.set_tags([0, 1])
    strategy = config()["strategy"]
    strategy["substrate"] = {"distance_tolerance": 0.1 if difference in ("substrate", "empty_substrate") else -1.0}
    population = SimpleNamespace(candidates=(first, second), similarity_counts={})
    observed = []

    class StopSelection(Exception):
        pass

    def inspect_pair(pool, *, compatible):
        observed.append(compatible(first, second))
        raise StopSelection

    manager = GeneticGenerationManager(
        strategy, PopulationConfig(config()["population"]), population,
        SimpleNamespace(select_pair=inspect_pair), np.random.default_rng(1))
    pairing = SimpleNamespace()
    if allow_variable is not None:
        pairing.allow_variable_composition = allow_variable
    operators = {"mobile": {"pairing": pairing, "mutations": SimpleNamespace(oplist=[])}}
    with pytest.raises(StopSelection):
        manager._reproduce(None, 1, population, operators, 0)
    assert observed == [allow_variable is True or difference in ("none", "empty_substrate")]


def test_compatible_pair_selection_weights_and_borrowing():
    frames = (candidate(1, score=2), candidate(2, score=3), candidate(3, "Ag2", score=20))
    population = SimpleNamespace(candidates=frames, similarity_counts={1: 3})
    calls = []

    class RecordChoices:
        def choice(self, indices, p):
            calls.append((list(indices), p.copy()))
            return indices[0]

    selector = GeneticParentSelector(RecordChoices())
    selector.participation = {2: 8}
    pair = selector.select_pair(population, compatible=lambda a, b: np.array_equal(a.numbers, b.numbers))
    assert pair[0] is frames[0] and pair[1] is frames[1]
    assert calls[0][0] == [0, 1]  # Isolated high-fitness candidate cannot be first.
    expected = compute_population_fitness(frames)[:2] / np.sqrt([1, 9]) / np.sqrt([4, 1])
    np.testing.assert_allclose(calls[0][1], expected / expected.sum())
    assert calls[1][0] == [1]
    calls.clear()
    selector.select_one(population)
    assert calls[0][0] == [0, 1, 2]  # Standalone mutation sees the entire pool.


def test_pair_selection_restart_and_metadata_immutability():
    frames = (candidate(1), candidate(2), candidate(3, "Ag2"))
    population = SimpleNamespace(candidates=frames, similarity_counts={})
    info = copy.deepcopy([a.info for a in frames])
    rng = np.random.default_rng(13)
    selector = GeneticParentSelector(rng)
    compatible = lambda a, b: np.array_equal(a.numbers, b.numbers)
    selector.select_pair(population, compatible=compatible)
    state = copy.deepcopy(rng.bit_generator.state)
    expected = [selector.select_pair(population, compatible=compatible)[0].info["confid"] for _ in range(20)]
    rng.bit_generator.state = state
    assert [selector.select_pair(population, compatible=compatible)[0].info["confid"] for _ in range(20)] == expected
    assert [a.info for a in frames] == info
