from types import SimpleNamespace

import pytest
from ase import Atoms

from gdpx.exploration import REGISTER
from gdpx.exploration.factory import create_exploration
from gdpx.exploration.genetic_algorithm.engine import (
    GeneticAlgorithmBroadcaster,
    GeneticAlgorithmEngine,
)
from gdpx.exploration.genetic_algorithm.generation import GeneticGenerationManager
from gdpx.exploration.genetic_algorithm.selection import GeneticParentSelector
from gdpx.exploration.population import Population
from gdpx.exploration.population.comparators import create_population_comparator
import numpy as np
from gdpx.exploration.persist.database import (
    CANDIDATES_DATABASE_FILENAME,
    GlobalOptimisationDatabase,
)
from gdpx.exploration.basin_hopping.engine import (
    BasinHopping,
)
from gdpx.exploration.population.config import PopulationConfig
from gdpx.exploration.monte_carlo.monte_carlo import MonteCarlo
from gdpx.exploration.sampling import parse_operators
from gdpx.exploration.simulated_annealing.simulated_annealing import SimulatedAnnealing


def make_generation_manager(params):
    GeneticGenerationManager.validate_parameters(params)
    config = PopulationConfig(params)
    population = Population(config.retained_size, create_population_comparator(config.comparator_config), config.use_extinct)
    rng = np.random.default_rng(7)
    selector = GeneticParentSelector(rng, params.get("name", "constant"))
    return GeneticGenerationManager(params, config, population, selector, rng)


def test_fragment_atom_order_is_canonical_without_changing_geometry():
    from gdpx.utils.atoms_tags import reassign_tags_by_species
    # The substrate keeps its original order. CO arrives as OC, with custom
    # per-atom data that must follow the corresponding physical atoms.
    atoms = Atoms('OCuOC', positions=np.arange(12).reshape(4, 3), tags=[0, 0, 7, 7])
    atoms.set_array('original_index', np.arange(4))
    atoms.info['nested'] = {'value': 1}
    reordered = reassign_tags_by_species(atoms)
    assert reordered.get_chemical_symbols() == ['O', 'Cu', 'C', 'O']
    np.testing.assert_array_equal(reordered.get_tags(), [0, 0, 1, 1])
    np.testing.assert_array_equal(reordered.arrays['original_index'], [0, 1, 3, 2])
    np.testing.assert_array_equal(reordered.positions, atoms.positions[[0, 1, 3, 2]])
    assert reordered.get_distance(2, 3) == atoms.get_distance(2, 3)
    np.testing.assert_array_equal(atoms.get_tags(), [0, 0, 7, 7])
    reordered.info['nested']['value'] = 2
    assert atoms.info['nested']['value'] == 1


def test_two_builder_co_example_has_compatible_fragment_order(tmp_path, monkeypatch):
    from pathlib import Path
    import yaml
    from gdpx.exploration.genetic_algorithm.core import RandomStreamRegistry
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    recipe = yaml.safe_load((root / 'examples/global_optimisation/explorations/genetic_algorithm/cu4_co_alumina111.yaml').read_text())['recipe']
    population = PopulationConfig(recipe['population'])
    builders = population.initialise_builders(recipe['population'], RandomStreamRegistry(recipe['random_seed']))
    for name, builder in builders.items():
        builder.directory = tmp_path / name
    frames = population._prepare_initial_population(builders)
    assert [frame.info['data']['builder'] for frame in frames] == ['random'] * 4 + ['site_insertion'] * 4
    assert all(len(frame) == 186 for frame in frames)
    assert all(np.array_equal(frame.numbers, frames[0].numbers) for frame in frames)
    assert all(np.array_equal(frame.get_tags(), frames[0].get_tags()) for frame in frames)


class Serializable:
    def __init__(self, value):
        self.value = value

    def as_dict(self):
        return self.value


class SerializableBuilder(Serializable):
    use_tags = False

    def set_rng(self, seed):
        self.random_seed = seed


@pytest.mark.parametrize(
    "method",
    [
        "genetic_algorithm",
        "monte_carlo",
        "basin_hopping",
        "simulated_annealing",
    ],
)
def test_factory_unpacks_recipe_and_keeps_seed(monkeypatch, method):
    captured = {}
    exploration = object()

    def create(**kwargs):
        captured.update(kwargs)
        return exploration

    monkeypatch.setitem(REGISTER._dict, method, create)

    result = create_exploration(
        {
            "method": method,
            "recipe": {
                "random_seed": 17,
                "operators": [],
                "convergence": {"steps": 2},
                **({"population": {}} if method == "basin_hopping" else {}),
            },
        }
    )

    assert result is exploration
    assert captured == {
        "random_seed": 17,
        "operators": [],
        "convergence": {"steps": 2},
        **({"population": {}} if method == "basin_hopping" else {}),
    }


@pytest.mark.parametrize(
    "config, message",
    [
        (
            {"method": "genetic_algorithm", "builder": {}, "params": {}},
            "requires a 'recipe' mapping",
        ),
        (
            {
                "method": "monte_carlo",
                "random_seed": 7,
                "recipe": {"operators": [], "convergence": {}},
            },
            "move these top-level keys into 'recipe': random_seed",
        ),
        ({"method": "simulated_annealing", "recipe": []}, "must be a mapping"),
    ],
)
def test_factory_rejects_legacy_global_optimisation_shapes(config, message):
    with pytest.raises((TypeError, ValueError), match=message):
        create_exploration(config)


def test_ga_broadcaster_uses_named_recipe_fields():
    broadcaster = GeneticAlgorithmBroadcaster(
        population={
            "periodic": False,
            "preserve_fragments": False,
            "builders": {"random": {"method": "unused"}},
            "initial": {"total_size": 1, "builder_allocations": [{"builder": "random", "size": 1}]},
            "generation": {
                "total_size": 1,
                "reproduction": {"size": 1},
                "mutation": {"size": 0},
                "completion": {"builder_proportions": [{"builder": "random", "proportion": 1.0}]},
            },
        },
        convergence={"generation": 1},
        objective={
            "target": "formation_energy",
            "chemical_potentials": {"Cu": [-3.0, -2.0]},
        },
        random_seed=23,
    )

    assert len(broadcaster.input_params_list) == 2
    assert {
        item["objective"]["chemical_potentials"]["Cu"]
        for item in broadcaster.input_params_list
    } == {-3.0, -2.0}
    assert all(item["random_seed"] == 23 for item in broadcaster.input_params_list)
    assert all("params" not in item for item in broadcaster.input_params_list)
    assert all("database" not in item for item in broadcaster.input_params_list)


def test_ga_omits_default_energy_objective():
    broadcaster = GeneticAlgorithmBroadcaster(
        population=_minimal_ga_population("random"),
        convergence={"generation": 1},
        objective={"target": "energy"},
        random_seed=23,
    )

    assert "objective" not in broadcaster.input_params_list[0]


def test_search_objective_rejects_legacy_keys():
    population = _minimal_ga_population("random")

    with pytest.raises(ValueError, match="property.*objective"):
        GeneticAlgorithmBroadcaster(
            population=population,
            convergence={"generation": 1},
            property={"target": "energy"},
        )

    with pytest.raises(ValueError, match="chempot.*chemical_potentials"):
        GeneticAlgorithmBroadcaster(
            population=population,
            convergence={"generation": 1},
            objective={"target": "formation_energy", "chempot": {"Cu": -3.0}},
        )

    with pytest.raises(ValueError, match="property.*objective"):
        BasinHopping(
            operators=[],
            num_mcmoves=1,
            mcworker={},
            population={},
            convergence={},
            property={"target": "energy"},
        )


def test_searches_reject_configurable_database_names():
    with pytest.raises(ValueError, match="database filename is no longer configurable"):
        GeneticAlgorithmBroadcaster(
            population=_minimal_ga_population("random"),
            convergence={"generation": 1},
            database="custom.db",
        )

    with pytest.raises(ValueError, match="population.database_fname is no longer configurable"):
        PopulationConfig({"database_fname": "custom.db"})


def test_population_searches_use_fixed_database_path(tmp_path):
    ga = object.__new__(GeneticAlgorithmEngine)
    ga.directory = tmp_path / "exploration-0"

    concurrent = object.__new__(BasinHopping)
    concurrent._directory = (tmp_path / "exploration-1").resolve()

    assert ga.db_path == (tmp_path / "exploration-0" / CANDIDATES_DATABASE_FILENAME).resolve()
    assert concurrent.database_path == (
        tmp_path / "exploration-1" / CANDIDATES_DATABASE_FILENAME
    ).resolve()
    assert ga.db_path != concurrent.database_path


def _minimal_ga_population(builder_name, reference_builder=None):
    population = {
        "periodic": False,
        "preserve_fragments": False,
        "builders": {builder_name: SerializableBuilder({"method": builder_name})},
        "initial": {
            "total_size": 1,
            "builder_allocations": [{"builder": builder_name, "size": 1}],
        },
        "generation": {
            "total_size": 1,
            "reproduction": {"size": 0},
            "mutation": {"size": 0},
            "completion": {
                "builder_proportions": [{"builder": builder_name, "proportion": 1.0}]
            },
        },
    }
    if reference_builder is not None:
        population["reference_builder"] = reference_builder
    return population


def test_ga_defaults_reference_builder_to_random():
    engine = GeneticAlgorithmEngine(
        population=_minimal_ga_population("random"),
        convergence={"generation": 1},
        random_seed=7,
    )

    assert engine.reference_builder_name == "random"
    assert engine.generator is engine.builders["random"]


def test_ga_requires_reference_when_random_builder_is_absent():
    with pytest.raises(ValueError, match="defaults reference_builder to 'random'"):
        GeneticAlgorithmEngine(
            population=_minimal_ga_population("alternative"),
            convergence={"generation": 1},
            random_seed=7,
        )

    engine = GeneticAlgorithmEngine(
        population=_minimal_ga_population("alternative", reference_builder="alternative"),
        convergence={"generation": 1},
        random_seed=7,
    )
    assert engine.reference_builder_name == "alternative"


def test_ga_population_uses_expanded_keys():
    population = make_generation_manager(
        {
            "periodic": False,
            "preserve_fragments": False,
            "initial": {
                "total_size": 4,
                "builder_allocations": [
                    {"builder": "imported", "size": 1},
                    {"builder": "random", "size": 3},
                ],
            },
            "generation": {
                "total_size": 4,
                "reproduction": {
                    "size": 2,
                    "maximum_attempts": 24,
                    "mutation_probability": 0.25,
                    "custom_mutation_probability": 0.75,
                },
                "mutation": {"size": 1, "maximum_attempts": 12},
                "completion": {
                    "builder_proportions": [
                        {"builder": "random", "proportion": 0.75},
                        {"builder": "alternative", "proportion": 0.25},
                    ]
                },
            },
            "substrate": {"distance_tolerance": 0.1},
        }
    )

    assert population.config.init_size == 4
    assert population.gen_rep_size == 2
    assert population.gen_mut_size == 1
    assert population.gen_mut_max_try == 12
    assert population.gen_rep_max_try == 24
    assert population.pmut == 0.25
    assert population.pmut_custom == 0.75
    assert population.substrate_dtol == 0.1
    assert population.allocate_completion_sizes(6) == [
        {"builder": "random", "size": 5, "maximum_attempts": 50},
        {"builder": "alternative", "size": 1, "maximum_attempts": 10},
    ]


def test_ga_population_defaults_system_description_to_true():
    base = {
        "initial": {
            "total_size": 1,
            "builder_allocations": [{"builder": "random", "size": 1}],
        },
        "generation": {
            "total_size": 1,
            "completion": {
                "builder_proportions": [{"builder": "random", "proportion": 1.0}]
            },
        },
    }

    population = PopulationConfig(base)
    assert population.periodic is True
    assert population.preserve_fragments is True

    disabled = PopulationConfig(
        dict(base, periodic=False, preserve_fragments=False)
    )
    assert disabled.periodic is False
    assert disabled.preserve_fragments is False

    mixed_periodic = dict(base, periodic=[True, True, False])
    with pytest.raises(ValueError, match="population.periodic must be a boolean"):
        PopulationConfig(mixed_periodic)

    invalid_fragments = dict(base, preserve_fragments="yes")
    with pytest.raises(ValueError, match="population.preserve_fragments must be a boolean"):
        PopulationConfig(invalid_fragments)


def test_population_candidate_validation_uses_system_description():
    config = {
        "periodic": True,
        "preserve_fragments": True,
        "initial": {
            "total_size": 1,
            "builder_allocations": [{"builder": "random", "size": 1}],
        },
        "generation": {
            "total_size": 1,
            "completion": {
                "builder_proportions": [{"builder": "random", "proportion": 1.0}]
            },
        },
    }
    population = PopulationConfig(config)

    tagged = Atoms("H", pbc=True)
    tagged.set_tags([1])
    population.validate_candidate(tagged, "test")

    with pytest.raises(ValueError, match="periodic boundary conditions"):
        population.validate_candidate(Atoms("H", pbc=False), "test")
    with pytest.raises(ValueError, match="no explicit ASE tags"):
        population.validate_candidate(Atoms("H", pbc=True), "test")
    substrate_only_tags = Atoms("H", pbc=True)
    substrate_only_tags.set_tags([0])
    with pytest.raises(ValueError, match="positive tags"):
        population.validate_candidate(substrate_only_tags, "test")


@pytest.mark.parametrize(
    ("key", "replacement"),
    [("pbc", "population.periodic"), ("use_tags", "population.preserve_fragments")],
)
def test_ga_rejects_population_owned_builder_keys(key, replacement):
    population = _minimal_ga_population("random")
    population["builders"]["random"] = {"method": "direct", key: False}

    with pytest.raises(ValueError, match=replacement):
        GeneticAlgorithmEngine(
            population=population,
            convergence={"generation": 1},
            random_seed=7,
        )


def test_ga_injects_default_system_settings_into_compatible_builders():
    population = _minimal_ga_population("random")
    population.pop("periodic")
    population.pop("preserve_fragments")
    population["builders"]["random"] = {
        "method": "random_structure_improved",
        "composition": {"Cu": 1},
        "box": [5.0, 5.0, 5.0],
    }

    engine = GeneticAlgorithmEngine(
        population=population,
        convergence={"generation": 1},
        random_seed=7,
    )

    assert engine.periodic is True
    assert engine.preserve_fragments is True
    assert engine.builders["random"].pbc is True
    assert engine.builders["random"].use_tags is True

    engine.worker = Serializable({"schema_version": 3})
    serialised_population = engine.as_dict()["recipe"]["population"]
    assert "periodic" not in serialised_population
    assert "preserve_fragments" not in serialised_population


def test_ga_fragment_policy_configures_and_rejects_operators():
    engine = object.__new__(GeneticAlgorithmEngine)
    engine.preserve_fragments = True
    compatible = SimpleNamespace(
        supports_fragment_preservation=True,
        fragment_mode_configurable=True,
        use_tags=False,
    )
    engine._configure_fragment_policy(compatible, "operators.mobile.mutation[0]")
    assert compatible.use_tags is True

    incompatible = SimpleNamespace(
        supports_fragment_preservation=False,
        fragment_mode_configurable=False,
    )
    with pytest.raises(ValueError, match="cannot guarantee"):
        engine._configure_fragment_policy(incompatible, "operators.mobile.mutation[1]")

    for key, value, replacement in (
        ("use_tags", True, "population.preserve_fragments"),
        ("pbc", True, "population.periodic"),
    ):
        with pytest.raises(ValueError, match=replacement):
            engine._reject_population_owned_operator_keys(
                {"method": "rattle", key: value}, "operators.mobile.mutation[0]"
            )


@pytest.mark.parametrize(
    "population",
    [
        {"init": {"size": 4}},
        {"generation": {"size": 4, "reprod": 4}},
        {
            "initial": {"total_size": 1, "builder_allocations": [{"builder": "a", "size": 1}]},
            "generation": {
                "total_size": 1,
                "completion": {"builder_proportions": [{"builder": "a", "proportion": 1.0}]},
            },
            "substrate": {"dtol": 0.1},
        },
    ],
)
def test_ga_population_rejects_abbreviated_keys(population):
    with pytest.raises(ValueError, match="Legacy GA"):
        make_generation_manager(population)


def test_monte_carlo_operator_probability_is_expanded():
    operators, weights = parse_operators(
        [
            {"method": "move", "particles": ["H"], "probability": 1.0},
            {"method": "move", "particles": ["H"], "probability": 3.0},
        ]
    )

    assert weights == [0.25, 0.75]
    assert operators[0].as_dict()["probability"] == 1.0
    with pytest.raises(ValueError, match="probability"):
        parse_operators([{"method": "move", "particles": ["H"], "prob": 1.0}])


def test_ga_serialization_uses_recipe_and_runtime():
    engine = object.__new__(GeneticAlgorithmEngine)
    engine.random_seed = 7
    engine.builders = {
        "random": Serializable(
            {"method": "random_structure_improved", "pbc": False, "use_tags": True}
        ),
        "imported": Serializable({"method": "direct", "use_tags": True}),
    }
    engine.reference_builder_name = "random"
    engine.worker = Serializable({"schema_version": 3})
    engine.ga_dict = {
        "population": {
            "periodic": False,
            "preserve_fragments": False,
            "initial": {"total_size": 1, "builder_allocations": [{"builder": "random", "size": 1}]},
            "generation": {"total_size": 1},
        },
        "operators": {},
        "convergence": {"generation": 1},
        "use_archive": True,
    }

    engine.population_config = PopulationConfig(engine.ga_dict["population"])
    engine.population_config.builders = engine.builders
    config = engine.as_dict()

    assert list(config) == ["method", "recipe", "runtime"]
    assert config["recipe"]["random_seed"] == 7
    assert config["recipe"]["population"]["builders"] == {
        "random": {"method": "random_structure_improved"},
        "imported": {"method": "direct"},
    }
    assert "reference_builder" not in config["recipe"]["population"]
    assert "params" not in config
    assert "worker" not in config
    assert "objective" not in config["recipe"]
    assert "database" not in config["recipe"]

    engine.reference_builder_name = "imported"
    assert engine.as_dict()["recipe"]["population"]["reference_builder"] == "imported"


class FixedBuilder:
    def __init__(self, symbol):
        self.symbol = symbol

    def run(self, size):
        return [Atoms(self.symbol) for _ in range(size)]


def test_initial_population_uses_ordered_builder_allocations():
    population = make_generation_manager(
        {
            "periodic": False,
            "preserve_fragments": False,
            "initial": {
                "total_size": 3,
                "builder_allocations": [
                    {"builder": "first", "size": 2},
                    {"builder": "second", "size": 1},
                ],
            },
            "generation": {
                "total_size": 3,
                "completion": {"builder_proportions": [{"builder": "first", "proportion": 1.0}]},
            },
        }
    )
    frames = population.config._prepare_initial_population(
        {"first": FixedBuilder("H"), "second": FixedBuilder("He")}
    )

    assert [atoms.get_chemical_formula() for atoms in frames] == ["H", "H", "He"]
    assert [atoms.info["data"]["builder"] for atoms in frames] == ["first", "first", "second"]


def test_generation_plan_round_trip(tmp_path):
    database = GlobalOptimisationDatabase(tmp_path / "ga.db")
    database.init_task(
        Atoms("H"),
        data={"population_size": 2, "initial_population_size": 2, "num_atoms_substrate": 1},
    )
    plan = {"stage": "completion", "completion_sizes": [{"builder": "random", "size": 2}]}

    database.set_generation_plan(1, plan)

    assert database.get_generation_plan(1) == plan


def test_generation_uses_reproduction_then_mutation_then_completion(tmp_path, monkeypatch):
    population = make_generation_manager(
        {
            "periodic": False,
            "preserve_fragments": False,
            "initial": {
                "total_size": 1,
                "builder_allocations": [{"builder": "first", "size": 1}],
            },
            "generation": {
                "total_size": 4,
                "reproduction": {"size": 2, "maximum_attempts": 1},
                "mutation": {"size": 1, "maximum_attempts": 1},
                "completion": {
                    "builder_proportions": [
                        {"builder": "first", "proportion": 0.5},
                        {"builder": "second", "proportion": 0.5},
                    ]
                },
            },
        }
    )
    population.selector = SimpleNamespace(select_one=lambda *args, **kwargs: Atoms("H"))
    monkeypatch.setattr(population, "_reproduce", lambda *args, **kwargs: None)

    class Mutation:
        def get_new_individual(self, parents):
            return Atoms("Li"), "mutation: direct"

    database = GlobalOptimisationDatabase(tmp_path / "stages.db")
    database.init_task(
        Atoms("H"),
        data={"population_size": 4, "initial_population_size": 1, "num_atoms_substrate": 1},
    )
    builders = {"first": FixedBuilder("He"), "second": FixedBuilder("Ne")}
    operators = {"mobile": {"mutations": Mutation()}}

    candidates = population._prepare_current_population(database, 1, builders, operators)

    assert [atoms.get_chemical_formula() for atoms in candidates] == ["Li", "He", "He", "Ne"]
    assert database.get_generation_plan(1)["stage"] == "complete"
    reloaded = population._get_current_candidates(database, 1)
    assert len(reloaded["mutated"]) == 1
    assert len(reloaded["completion"]) == 3
    assert len(population._prepare_current_population(database, 1, builders, operators, reloaded)) == 4


def test_monte_carlo_serialization_uses_recipe_and_runtime():
    engine = object.__new__(MonteCarlo)
    engine.random_seed = 11
    engine.builder = Serializable({"method": "builder"})
    engine.worker = Serializable({"schema_version": 3})
    engine.worker.runtime = SimpleNamespace(provider_potential=object())
    engine.operators = [Serializable({"method": "move"})]
    engine.convergence = {"steps": 5}
    engine.dump_period = 2
    engine.ckpt_period = 10
    engine.ignore_atoms_tags = True
    engine.should_retry = False
    engine.restart = False

    config = engine.as_dict()

    assert list(config) == ["method", "recipe", "runtime"]
    assert config["recipe"]["random_seed"] == 11
    assert config["recipe"]["operators"] == [{"method": "move"}]


def test_other_global_optimisers_serialize_the_recipe():
    worker = Serializable({"schema_version": 3})

    concurrent = object.__new__(BasinHopping)
    concurrent.random_seed = 13
    concurrent.worker = worker
    concurrent._init_params = {
        "population": _minimal_ga_population("random"),
        "convergence": {"generation": 2},
        "objective": {
            "target": "formation_energy",
            "chemical_potentials": {"O": -4.95},
        },
        "use_archive": True,
    }
    concurrent.population_config = PopulationConfig(concurrent._init_params["population"])
    concurrent.population_config.builders = concurrent._init_params["population"]["builders"]
    concurrent_config = concurrent.as_dict()
    assert concurrent_config["method"] == "basin_hopping"

    annealing = object.__new__(SimulatedAnnealing)
    annealing.random_seed = 19
    annealing.worker = worker
    annealing.builder = Serializable({"method": "builder"})
    annealing.temperatures = [800.0, 400.0]
    annealing_config = annealing.as_dict()

    assert concurrent_config["recipe"]["random_seed"] == 13
    assert concurrent_config["recipe"]["population"]["retained_size"] == 1
    assert concurrent_config["recipe"]["objective"] == {
        "target": "formation_energy",
        "chemical_potentials": {"O": -4.95},
    }
    assert annealing_config == {
        "method": "simulated_annealing",
        "recipe": {
            "random_seed": 19,
            "builder": {"method": "builder"},
            "temperatures": [800.0, 400.0],
        },
        "runtime": {"schema_version": 3},
    }


def test_basin_hopping_replaces_the_old_alias_and_registration():
    from gdpx.exploration.exploration import BaseExploration
    assert REGISTER["basin_hopping"] is BasinHopping
    assert BasinHopping.__bases__ == (BaseExploration,)
    assert "concurrent_hopping" not in REGISTER
    with pytest.raises(ValueError, match="renamed to basin_hopping"):
        create_exploration({"method": "concurrent_hopping", "recipe": {}})
    with pytest.raises(ValueError, match="former MC alias use method: monte_carlo"):
        create_exploration({"method": "basin_hopping", "recipe": {"convergence": {"steps": 2}}})


def test_basin_hopping_accepts_a_constructed_cli_worker():
    from gdpx.execution.workers.single import SingleWorker

    engine = object.__new__(BasinHopping)
    worker = object.__new__(SingleWorker)
    engine.register_worker(worker)
    assert engine.worker is worker
