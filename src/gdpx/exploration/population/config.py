"""Shared population settings and initial candidate construction."""
import copy
import inspect
from typing import Optional
from collections.abc import Mapping

import numpy as np
from ase import Atoms

from gdpx.structures.builders import REGISTER as BUILDER_REGISTER
from gdpx.structures.builders.factory import canonicalise_builder
from ..persist.thanos import dispatch_thanos


def clean_seed_structures(frames):
    """Create independently owned initial candidates without losing custom arrays."""
    cleaned = []
    for frame in frames:
        atoms = frame.copy()
        atoms.info = {}
        cleaned.append(atoms)
    return cleaned


class PopulationConfig:
    """Settings shared by BH and GA; algorithm policies extend generation settings."""

    MAX_ATTEMPTS_MULTIPLIER = 10
    _print = staticmethod(print)
    _debug = staticmethod(print)

    def __init__(self, params, rng=None):
        if not isinstance(params, Mapping):
            raise ValueError("population must be a mapping.")
        if "database_fname" in params:
            raise ValueError("population.database_fname is no longer configurable; remove it.")
        replacements = {
            "initial_size": "initial.total_size",
            "generation_size": "generation.total_size",
            "population_size": "retained_size",
            "random_offspring_generator": "builders and initial.builder_allocations",
        }
        old = replacements.keys() & params.keys()
        if old:
            migration = ", ".join(f"{key} -> {replacements[key]}" for key in sorted(old))
            raise ValueError(f"Legacy population keys are not supported: {migration}.")
        self.rng = np.random.default_rng() if rng is None else rng
        for section in ("initial", "generation"):
            if not isinstance(params.get(section), Mapping):
                raise ValueError(f"population.{section} must be a mapping.")
        self.init_size = self._positive_integer(params["initial"].get("total_size"), "initial.total_size")
        self.gen_size = self._positive_integer(params["generation"].get("total_size"), "generation.total_size")
        self.retained_size = self._positive_integer(params.get("retained_size", self.gen_size), "population.retained_size")
        self.initial_builder_allocations = self._parse_builder_allocations(
            params["initial"].get("builder_allocations"), self.init_size
        )
        self.periodic = self._boolean_setting(params, "periodic", True)
        self.preserve_fragments = self._boolean_setting(params, "preserve_fragments", True)
        self.comparator_config = copy.deepcopy(params.get("comparator", {"method": "interatomic_distance"}))
        thanos = params.get("thanos")
        self.extinct_callbacks = None
        if thanos is not None:
            configs = [thanos] if isinstance(thanos, Mapping) else thanos
            self.extinct_callbacks = [dispatch_thanos(**copy.deepcopy(c)) for c in configs]
        self.use_extinct = self.extinct_callbacks is not None

    def initialise_builders(self, params, streams):
        configs = params.get("builders")
        if not isinstance(configs, Mapping) or not configs:
            raise ValueError("Population configuration requires a non-empty 'builders' mapping.")
        if not all(isinstance(name, str) and name for name in configs):
            raise ValueError("Population builder names must be non-empty strings.")
        reference = params.get("reference_builder", "random")
        if not isinstance(reference, str) or reference not in configs:
            raise ValueError(
                "population.reference_builder must name one of population.builders; "
                "GDPy defaults reference_builder to 'random'."
            )
        self.reference_builder_name = reference
        self.builders = {}
        for name, config in configs.items():
            seed = streams.seed(f"builder/{name}")
            if isinstance(config, Mapping):
                config = copy.deepcopy(dict(config))
                old = {"pbc", "use_tags"} & config.keys()
                if old:
                    raise ValueError(
                        f"Population builder {name!r} contains population-owned keys: "
                        "pbc -> population.periodic, use_tags -> population.preserve_fragments."
                    )
                builder_class = BUILDER_REGISTER[config.get("method", "direct")]
                if "pbc" in inspect.signature(builder_class.__init__).parameters:
                    config["pbc"] = self.periodic
                config["random_seed"] = seed
                builder = canonicalise_builder(config)
            else:
                builder = config
                if not hasattr(builder, "set_rng"):
                    raise TypeError(f"Population builder {name!r} must be a builder configuration or instance.")
                builder.set_rng(seed)
            if builder is None:
                raise ValueError(f"Population builder {name!r} could not be initialised.")
            if hasattr(builder, "use_tags"):
                builder.use_tags = True
            builder.rng = streams.get(f"builder/{name}")
            self.builders[name] = builder
        self._require_builders(self.builders, [a["builder"] for a in self.initial_builder_allocations])
        return self.builders

    def serialise(self, params):
        result = {key: copy.deepcopy(value) for key, value in params.items() if key != "builders"}
        result["retained_size"] = self.retained_size
        result["comparator"] = copy.deepcopy(self.comparator_config)
        result["builders"] = {}
        for name, builder in self.builders.items():
            config = copy.deepcopy(builder.as_dict())
            config.pop("pbc", None)
            config.pop("use_tags", None)
            result["builders"][name] = config
        return result

    @staticmethod
    def _boolean_setting(params: Mapping, key: str, default: bool) -> bool:
        value = params.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"population.{key} must be a boolean; got {value!r}.")
        return value

    @staticmethod
    def _positive_integer(value, path: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{path} must be a positive integer; got {value!r}.")
        return value

    @staticmethod
    def _nonnegative_integer(value, path: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{path} must be a non-negative integer; got {value!r}.")
        return value

    def _parse_builder_allocations(self, allocations, total_size: int) -> list[dict]:
        if not isinstance(allocations, list) or not allocations:
            raise ValueError("initial.builder_allocations must be a non-empty list.")
        parsed = []
        names = set()
        for index, allocation in enumerate(allocations):
            if (
                not isinstance(allocation, Mapping)
                or not isinstance(allocation.get("builder"), str)
                or not allocation["builder"]
            ):
                raise ValueError(f"initial.builder_allocations[{index}] requires a builder name.")
            size = self._nonnegative_integer(allocation.get("size"), f"initial.builder_allocations[{index}].size")
            maximum_attempts = allocation.get("maximum_attempts", size * self.MAX_ATTEMPTS_MULTIPLIER)
            maximum_attempts = self._nonnegative_integer(
                maximum_attempts, f"initial.builder_allocations[{index}].maximum_attempts"
            )
            name = allocation["builder"]
            if name in names:
                raise ValueError(f"initial.builder_allocations repeats builder {name!r}.")
            names.add(name)
            parsed.append(dict(builder=name, size=size, maximum_attempts=maximum_attempts))
        if sum(x["size"] for x in parsed) != total_size:
            raise ValueError("initial builder allocation sizes must sum to initial.total_size.")
        return parsed

    def _generate_from_builder(self, name: str, builder, size: int, maximum_attempts: int) -> list[Atoms]:
        frames: list[Atoms] = []
        for _ in range(maximum_attempts):
            if len(frames) == size:
                break
            generated = builder.run(size=size - len(frames))
            if isinstance(generated, Atoms):
                generated = [generated]
            if generated is None:
                generated = []
            if not isinstance(generated, list) or not all(isinstance(atoms, Atoms) for atoms in generated):
                raise RuntimeError(f"Builder {name!r} returned invalid structures.")
            if len(generated) > size - len(frames):
                raise RuntimeError(f"Builder {name!r} returned more structures than requested.")
            for offset, atoms in enumerate(generated):
                self.validate_candidate(atoms, f"builder {name!r}", len(frames) + offset)
            frames.extend(generated)
        if len(frames) != size:
            raise RuntimeError(
                f"Builder {name!r} generated {len(frames)} of {size} requested structures "
                f"after {maximum_attempts} attempts."
            )
        return frames

    def validate_candidate(self, atoms: Atoms, source: str, index: Optional[int] = None) -> None:
        """Validate population-wide system invariants for a candidate."""
        location = source if index is None else f"{source} candidate {index}"
        expected_pbc = np.full(3, self.periodic, dtype=bool)
        if not np.array_equal(atoms.get_pbc(), expected_pbc):
            raise ValueError(
                f"{location} has periodic boundary conditions {atoms.get_pbc().tolist()}, "
                f"but population.periodic is {self.periodic}."
            )
        if self.preserve_fragments and not atoms.has("tags"):
            raise ValueError(
                f"{location} has no explicit ASE tags, but population.preserve_fragments is true."
            )
        if self.preserve_fragments:
            tags = atoms.get_tags()
            if np.any(tags < 0) or not np.any(tags > 0):
                raise ValueError(
                    f"{location} must use tag 0 only for substrate atoms and positive tags "
                    "for mobile fragments when population.preserve_fragments is true."
                )

    @staticmethod
    def _require_builders(builders: Mapping, names) -> None:
        missing = sorted(set(names) - set(builders))
        if missing:
            raise ValueError(f"Unknown population builders: {', '.join(missing)}.")

    def _prepare_initial_population(self, builders: Mapping) -> list[Atoms]:
        """Build the initial population from explicit, strictly sized allocations."""
        self._require_builders(builders, (x["builder"] for x in self.initial_builder_allocations))
        starting_population = []
        for allocation in self.initial_builder_allocations:
            name = allocation["builder"]
            frames = self._generate_from_builder(
                name, builders[name], allocation["size"], allocation["maximum_attempts"]
            )
            starting_population.extend(self.clean_initial_structures(frames, name))
        if len(starting_population) != self.init_size:
            raise RuntimeError("Failed to generate the configured initial population.")
        return starting_population

    @staticmethod
    def clean_initial_structures(frames: list[Atoms], builder_name: str) -> list[Atoms]:
        """Remove calculators and attach persisted initial-builder metadata."""
        cleaned = clean_seed_structures(frames)
        for atoms in cleaned:
            atoms.info["data"] = {"builder": builder_name}
            atoms.info["key_value_pairs"] = dict(
                origin=f"InitialBuilder:{builder_name}", extinct=0
            )
        return cleaned

