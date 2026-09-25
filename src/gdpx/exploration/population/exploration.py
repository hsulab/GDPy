"""Shared construction and configuration of population-based searches."""
import copy
from collections.abc import Mapping

from ..exploration import BaseExploration
from .config import PopulationConfig
from .population import Population
from .comparators import create_population_comparator
from .random import RandomStreamRegistry


SHARED_STRATEGY_SETTINGS = ("objective", "convergence", "use_archive")


def validate_strategy(strategy, method=None):
    if not isinstance(strategy, Mapping):
        raise ValueError("global_optimisation requires a strategy mapping with strategy.method.")
    selected = strategy.get("method")
    allowed = {
        "genetic_algorithm": {"operators", "reproduction", "mutation", "completion", "substrate"},
        "basin_hopping": {"operators", "steps_per_chain", "selection"},
    }
    if not isinstance(selected, str) or selected not in allowed:
        raise ValueError("strategy.method must be genetic_algorithm or basin_hopping.")
    if method is not None and selected != method:
        raise ValueError(f"Expected strategy.method: {method}, got {selected!r}.")
    if "num_mcmoves" in strategy:
        raise ValueError("strategy.num_mcmoves was renamed to strategy.steps_per_chain.")
    unknown = strategy.keys() - allowed[selected] - {"method"}
    if unknown:
        raise ValueError(
            f"Unsupported {selected} strategy settings: {', '.join(sorted(unknown))}. "
            "Crossover parent compatibility is determined automatically by the operator."
        )
    operators = strategy.get("operators")
    if selected == "genetic_algorithm" and operators is not None and not isinstance(operators, Mapping):
        raise ValueError("strategy.operators must be a mapping for genetic algorithms.")
    return copy.deepcopy(dict(strategy))


def split_global_strategy(strategy):
    """Separate public shared settings from one algorithm strategy."""
    if not isinstance(strategy, Mapping):
        raise ValueError("global_optimisation requires a strategy mapping with strategy.method.")
    public = copy.deepcopy(dict(strategy))
    shared = {key: public.pop(key) for key in SHARED_STRATEGY_SETTINGS if key in public}
    return validate_strategy(public), shared


def reject_legacy_settings(parameters):
    moved_sections = {
        "population": "system",
        "objective": "strategy.objective",
        "convergence": "strategy.convergence",
        "use_archive": "strategy.use_archive",
    }
    moved_top_level = moved_sections.keys() & parameters.keys()
    if moved_top_level:
        raise ValueError(
            "Move global_optimisation settings into the three-section layout: "
            + ", ".join(
                f"{key} -> {moved_sections[key]}" for key in sorted(moved_top_level)
            )
        )
    if "num_mcmoves" in parameters:
        raise ValueError("num_mcmoves moved to strategy.steps_per_chain.")
    if "mcworker" in parameters:
        raise ValueError("BH mcworker was removed; move calculation settings into top-level runtime.")
    if "builder" in parameters:
        raise ValueError("Use system.builders and system.initial.builder_allocations instead of builder.")
    if "recipe" in parameters:
        raise ValueError(
            "global_optimisation has no recipe wrapper; move population construction "
            "under system and search settings under strategy."
        )
    moved = {"operators", "steps_per_chain", "selection", "reproduction", "mutation", "completion", "substrate"}
    moved &= parameters.keys()
    if moved:
        raise ValueError(
            "Move search settings under strategy: "
            + ", ".join(f"{key} -> strategy.{key}" for key in sorted(moved))
        )


def create_global_optimisation(
    system, strategy, random_seed=None,
):
    """Dispatch a three-section global-optimisation configuration."""
    strategy, shared = split_global_strategy(strategy)
    convergence = shared.get("convergence")
    common = dict(
        population=system, strategy=strategy, objective=shared.get("objective"),
        random_seed=random_seed, use_archive=shared.get("use_archive", True),
    )
    if strategy["method"] == "genetic_algorithm":
        from ..genetic_algorithm.engine import GeneticAlgorithmBroadcaster

        if convergence is None:
            raise ValueError("Genetic algorithm requires convergence settings.")
        return GeneticAlgorithmBroadcaster(convergence=convergence, **common)
    from ..basin_hopping.engine import BasinHopping

    return BasinHopping(convergence=convergence, **common)


class PopulationBasedExploration(BaseExploration):
    """Common population setup; subclasses own selection and execution."""

    def __init__(self, population, strategy, *args, **kwargs):
        reject_legacy_settings(kwargs)
        super().__init__(*args, **kwargs)
        self.strategy_config = validate_strategy(strategy)
        self.random_streams = RandomStreamRegistry(self.random_seed)
        self.rng = self.random_streams.get("engine")
        self.population_config = PopulationConfig(population, rng=self.random_streams.get("population"))
        self.periodic = self.population_config.periodic
        self.preserve_fragments = self.population_config.preserve_fragments
        self.population_comparator = create_population_comparator(
            self.population_config.comparator_config, self.periodic,
            self.random_streams.get("population/comparator"),
        )
        self.population = Population(
            self.population_config.retained_size,
            self.population_comparator, self.population_config.use_extinct,
        )
        self._configure_population_strategy()
        self.builders = self.population_config.initialise_builders(population, self.random_streams)
        self.reference_builder_name = self.population_config.reference_builder_name
        self.generator = self.builders[self.reference_builder_name]
        self.worker = None

    def _configure_population_strategy(self):
        """Validate population-dependent policies before constructing builders."""

    def serialise_search(self, parameters):
        result = copy.deepcopy(parameters)
        population = result.pop("population")
        strategy = result.pop("strategy")
        for key in SHARED_STRATEGY_SETTINGS:
            if key in result:
                strategy[key] = result.pop(key)
        system = self.population_config.serialise(population)
        reference = getattr(self, "reference_builder_name", "random")
        if reference != "random":
            system["reference_builder"] = reference
        return dict(
            method="global_optimisation", random_seed=self.random_seed,
            system=system, strategy=strategy, **result,
            runtime=copy.deepcopy(self.worker.as_dict()),
        )
