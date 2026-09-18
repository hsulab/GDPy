import copy
from typing import Callable, Mapping, Optional

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from gdpx.exploration.persist.database import GlobalOptimisationDatabase as GODB
from gdpx.exploration.persist.thanos import dispatch_thanos
from gdpx.utils.atoms_tags import get_tags_per_species
from gdpx.utils.profiler import CustomTimer

from .population import Population, PopulationWithVariableComposition

#: Retained keys in key_value_pairs when get_atoms from the database.
RETAINED_KEYS: list[str] = ["extinct", "origin"]


def clean_seed_structures(prev_frames: list[Atoms]) -> list[Atoms]:
    """"""
    curr_frames = []
    # energies, forces = [], []
    for _, prev_atoms in enumerate(prev_frames):
        # copy geometry
        curr_atoms = Atoms(
            symbols=copy.deepcopy(prev_atoms.get_chemical_symbols()),
            positions=copy.deepcopy(prev_atoms.get_positions()),
            cell=copy.deepcopy(prev_atoms.get_cell(complete=True)),
            pbc=copy.deepcopy(prev_atoms.get_pbc()),
            tags=prev_atoms.get_tags(),  # retain this for molecules
        )
        # if prev_atoms.get_kinetic_energy() > 0.: # retain this for MD
        #    curr_atoms.set_momenta(prev_atoms.get_momenta())
        curr_frames.append(curr_atoms)

        # save properties
        # try:
        #     ene = prev_atoms.get_potential_energy()
        #     energies.append(ene)
        # except:
        #     raise RuntimeError(f"Cannot get energy for seed structure {i}.")
        #
        # try:
        #     frc = prev_atoms.get_forces()
        #     forces.append(frc)
        # except:
        #     raise RuntimeError(f"Cannot get forces for seed structure {i}.")

    # for a, e, f in zip(curr_frames, energies, forces):
    #     calc = SinglePointCalculator(a, energy=e, forces=f)
    #     a.calc = calc

    return curr_frames


def compare_two_atoms_by_substrates(a0: Atoms, a1: Atoms, dtol: float = 0.20) -> bool:
    """Compare two atoms by substrates.

    Args:
        a0: The first structure.
        a1: The second structure.
        dtol: The distance tolerance.

    """
    similar = True

    # Find the substrate in the atoms, which always has zero tags.
    a0_substrate_indices = [i for i, tag in enumerate(a0.get_tags()) if tag == 0]
    num_a0_substrate = len(a0_substrate_indices)

    a1_substrate_indices = [i for i, tag in enumerate(a1.get_tags()) if tag == 0]
    num_a1_substrate = len(a1_substrate_indices)

    if num_a0_substrate == num_a1_substrate:
        # Check the positions of the substrate atoms. The cell should be checked before.
        a0_substrate_positions = a0.get_positions()[a0_substrate_indices]
        a1_substrate_positions = a1.get_positions()[a1_substrate_indices]
        _, mic_distances = find_mic(a1_substrate_positions - a0_substrate_positions, a0.get_cell())
        dmax = np.max(mic_distances)
        a0.info["dmax"] = dmax
        if dmax <= dtol:
            similar = True
        else:
            similar = False
    else:
        similar = False

    return similar


def is_reproduction_isolation(candidates: Optional[tuple[Atoms, Atoms]]) -> bool:
    """"""
    is_isolation = True
    if candidates is not None:
        a0, a1 = candidates
        natoms_a0, natoms_a1 = len(a0), len(a1)
        if natoms_a0 == natoms_a1:
            symbols_a0, symbols_a1 = (
                a0.get_chemical_symbols(),
                a1.get_chemical_symbols(),
            )
            if symbols_a0 == symbols_a1:
                if np.array_equal(a0.get_tags(), a1.get_tags()):
                    is_isolation = False
    else:
        is_isolation = True

    return is_isolation


def extinct_candidate(atoms: Atoms, extinct_callbacks: list[Callable]) -> None:
    """Extinct the candidate by given callback.

    Args:
        atoms: The candidate to be evaluated.
        extinct_callbacks: A list of callback functions that determine extinction using 0 or 1.

    """
    extincts = [cb(atoms) for cb in extinct_callbacks]

    extinct = int(sum(extincts) > 0)
    atoms.info["key_value_pairs"]["extinct"] = extinct

    return


class PopulationManager:
    """An abstract population manager for evolutionary algorithms.

    For structure exploration, there are generally two formulations. ASE forms
    current population from all candidates while USPEX forms it based on the
    previous generation. Furthermore, USPEX uses fracGene, fracRand, fracTopRand,
    fracPerm, fracAtomsMut, fracRotMut, fracLatMut, fracSpinMut...

    Example:
        Parameters needed

        $ cat ga.yaml
        population:
            builders:
                random:
                    method: random_structure_improved
            initial:
                total_size: 50
                builder_allocations:
                  - builder: random
                    size: 50
            generation:
                total_size: 20
                reproduction:
                    size: 16
                    mutation_probability: 0.5
                mutation:
                    size: 2
                completion:
                    builder_proportions:
                      - builder: random
                        proportion: 1.0

    """

    _print = print

    _debug = print

    #: Maximum attempts to generate new structures.
    MAX_ATTEMPTS_MULTIPLIER: int = 10

    def __init__(self, params: dict, rng=np.random.default_rng()) -> None:
        """"""
        self.rng = rng

        legacy_keys = {"init", "gen", "pmut", "pmut_custom", "random_generator", "reproduction"}.intersection(
            params
        )
        if legacy_keys:
            replacements = {
                "init": "initial",
                "gen": "generation",
                "pmut": "generation.reproduction.mutation_probability",
                "pmut_custom": "generation.reproduction.custom_mutation_probability",
                "random_generator": "builders",
                "reproduction": "generation.reproduction",
            }
            migration = ", ".join(f"{key} -> {replacements[key]}" for key in sorted(legacy_keys))
            raise ValueError(f"Legacy GA population keys are not supported: {migration}.")

        # Get population name
        name = params.get("name", "constant")
        if name not in ["constant", "variable"]:
            raise Exception("Population name must be `constant` or `variable`.")
        self.name = name

        gen_params = params.get("generation", {})
        if not isinstance(gen_params, Mapping):
            raise ValueError("population.generation must be a mapping.")
        legacy_generation_keys = {
            "reprod": "reproduction",
            "mutate": "mutation",
            "max_random_try": "reproduction.maximum_attempts",
            "max_reprod_try": "reproduction.maximum_attempts",
            "size": "total_size",
            "random": "completion.builder_proportions",
        }
        found_legacy_generation_keys = legacy_generation_keys.keys() & gen_params.keys()
        if found_legacy_generation_keys:
            migration = ", ".join(
                f"{key} -> {legacy_generation_keys[key]}" for key in sorted(found_legacy_generation_keys)
            )
            raise ValueError(f"Legacy GA generation keys are not supported: {migration}.")

        init_params = params.get("initial", {})
        if not isinstance(init_params, Mapping):
            raise ValueError("population.initial must be a mapping.")
        rejected_initial = {"size", "seed_file", "sources", "fallback_builder"}.intersection(init_params)
        if rejected_initial:
            raise ValueError(
                "Legacy GA initial keys are not supported: " + ", ".join(sorted(rejected_initial)) + "."
            )
        self.init_size = self._positive_integer(init_params.get("total_size"), "initial.total_size")
        self.initial_builder_allocations = self._parse_builder_allocations(
            init_params.get("builder_allocations"), self.init_size
        )

        # Get number of structures from different origins in one generation
        self.gen_size = self._positive_integer(gen_params.get("total_size"), "generation.total_size")
        reproduction_params = gen_params.get("reproduction", {})
        mutation_params = gen_params.get("mutation", {})
        completion_params = gen_params.get("completion", {})
        generation_sections = (reproduction_params, mutation_params, completion_params)
        if not all(isinstance(section, Mapping) for section in generation_sections):
            raise ValueError(
                "generation.reproduction, generation.mutation, and generation.completion must be mappings."
            )
        self.gen_rep_size = self._nonnegative_integer(reproduction_params.get("size", 0), "reproduction.size")
        self.gen_mut_size = self._nonnegative_integer(mutation_params.get("size", 0), "mutation.size")
        if self.gen_rep_size + self.gen_mut_size > self.gen_size:
            raise ValueError("generation reproduction and mutation sizes exceed generation.total_size.")
        self.gen_rep_max_try = self._attempts(reproduction_params, self.gen_rep_size, "reproduction")
        self.gen_mut_max_try = self._attempts(mutation_params, self.gen_mut_size, "mutation")
        self.completion_builder_proportions = self._parse_builder_proportions(
            completion_params.get("builder_proportions")
        )

        # Mutation probabilities for offspring produced by reproduction.
        self.pmut = reproduction_params.get("mutation_probability", 0.5)
        self.pmut_custom = reproduction_params.get("custom_mutation_probability", 0.5)

        # Get the tolerance for comparing two atoms by substrates
        substrate_params = params.get("substrate", dict(distance_tolerance=-1.0))
        if "dtol" in substrate_params:
            raise ValueError("Legacy GA substrate key 'dtol' is not supported; use 'distance_tolerance'.")
        self.substrate_dtol = substrate_params.get("distance_tolerance", -1.0)  # Ang

        # Thanos (observer/describer) extincts structures in the population
        thanos = params.get("thanos", None)
        extinct_callbacks = None
        if thanos is not None:
            thanos_config = copy.deepcopy(thanos)
            # check whether dict or list by mapping
            if isinstance(thanos_config, Mapping):
                thanos_config = [thanos_config]
            extinct_callbacks = [dispatch_thanos(**tc) for tc in thanos_config]
        else:
            ...

        self.extinct_callbacks = extinct_callbacks
        self.use_extinct = True if extinct_callbacks is not None else False

        # Lazy attributes
        self.population = None

        return

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

    def _attempts(self, params: Mapping, size: int, section: str) -> int:
        attempts = params.get("maximum_attempts", size * self.MAX_ATTEMPTS_MULTIPLIER)
        return self._nonnegative_integer(attempts, f"generation.{section}.maximum_attempts")

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

    def _parse_builder_proportions(self, proportions) -> list[dict]:
        if not isinstance(proportions, list) or not proportions:
            raise ValueError("generation.completion.builder_proportions must be a non-empty list.")
        parsed = []
        names = set()
        for index, item in enumerate(proportions):
            if not isinstance(item, Mapping) or not isinstance(item.get("builder"), str) or not item["builder"]:
                raise ValueError(f"generation.completion.builder_proportions[{index}] requires a builder name.")
            proportion = item.get("proportion")
            if not isinstance(proportion, (int, float)) or isinstance(proportion, bool) or proportion <= 0:
                raise ValueError(f"generation completion proportion at index {index} must be positive.")
            name = item["builder"]
            if name in names:
                raise ValueError(f"generation completion proportions repeat builder {name!r}.")
            names.add(name)
            maximum_attempts = item.get("maximum_attempts")
            if maximum_attempts is not None:
                maximum_attempts = self._nonnegative_integer(
                    maximum_attempts,
                    f"generation.completion.builder_proportions[{index}].maximum_attempts",
                )
            parsed.append(
                dict(builder=name, proportion=float(proportion), maximum_attempts=maximum_attempts)
            )
        if not np.isclose(sum(x["proportion"] for x in parsed), 1.0):
            raise ValueError("generation completion builder proportions must sum to 1.0.")
        return parsed

    def allocate_completion_sizes(self, size: int) -> list[dict]:
        """Allocate a deficit using largest remainder and declaration-order ties."""
        raw = [size * item["proportion"] for item in self.completion_builder_proportions]
        allocated = [int(np.floor(value)) for value in raw]
        order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - allocated[i]), i))
        for index in order[: size - sum(allocated)]:
            allocated[index] += 1
        return [
            dict(
                builder=item["builder"],
                size=item_size,
                maximum_attempts=(
                    item["maximum_attempts"]
                    if item["maximum_attempts"] is not None
                    else item_size * self.MAX_ATTEMPTS_MULTIPLIER
                ),
            )
            for item, item_size in zip(self.completion_builder_proportions, allocated)
        ]

    def update_population(self, database: GODB, comparing) -> None:
        """Update population.

        Args:
            database: GODB.
            comparing: A comparing operator.

        """
        if self.name == "constant":
            population = Population(
                data_connection=database,
                population_size=self.gen_size,
                comparator=comparing,
                use_extinct=self.use_extinct,
                rng=self.rng,
                print_func=self._print,
                debug_func=self._debug,
            )
            # self._print(f"population number: {len(current_population.pop)}")
        elif self.name == "variable":
            population = PopulationWithVariableComposition(
                data_connection=database,
                population_size=self.gen_size,
                comparator=comparing,
                use_extinct=self.use_extinct,
                rng=self.rng,
                print_func=self._print,
                debug_func=self._debug,
            )
            # for tribe in population.tribes:
            #     self._print(f"tribe: {tribe[0]} number: {len(tribe[1])}")
        else:
            raise RuntimeError(f"Population name `{self.name}` is not supported.")

        self.population = population

        return

    def _extinct_candidate(self, atoms: Atoms) -> None:
        """Extinct the candidate by given callback.

        Args:
            atoms: The candidate to be evaluated.

        """
        if self.use_extinct and self.extinct_callbacks is not None:
            extinct_candidate(atoms, self.extinct_callbacks)

        return

    def _get_current_candidates(self, database: GODB, curr_gen: int) -> dict[str, list[Atoms]]:
        """Get offsprings in the current generation.

        Mutataed candidates do not have `generation` keyword.

        Args:
            database: GODB.
            curr_gen: The current generation number.

        """
        candidate_groups = {"paired": [], "mutated": [], "completion": [], "initial": []}

        with CustomTimer(name="getting canidates in the current generation", func=self._print):
            unrelaxed_candidate_rows = list(database.connection.select(f"relaxed=0,generation={curr_gen}"))
        for row in unrelaxed_candidate_rows:
            if row.formula:
                confid = row["confid"]
                curr_rows = sorted(
                    database.connection.select(f"relaxed=0,confid={confid}"),
                    key=lambda x: x.mtime,
                )
                curr_rows = [x for x in curr_rows if x.formula]
                # get latest atoms, if pairing+mutation, the latest atoms should be the mutated one
                curr_atoms = database.connection.get_atoms(curr_rows[-1].id, add_additional_information=True)
                # Keep only candidate state needed after reloading; database
                # event-classification fields belong to the stored rows.
                kvp = {k: v for k, v in curr_atoms.info["key_value_pairs"].items() if k in RETAINED_KEYS}
                data = curr_atoms.info.get("data", {})  # not every cand has data that stores parents
                curr_atoms.info = {
                    "key_value_pairs": kvp,
                    "data": data,
                    "confid": confid,
                }
                # we use the first row to determine the origin as the pairing may be followed by a mutation
                # but it should be considered still from the pairing.
                origin = curr_rows[0]["origin"]
                if "Pairing" in origin:
                    candidate_groups["paired"].append(curr_atoms)
                elif "Mutation" in origin:
                    candidate_groups["mutated"].append(curr_atoms)
                elif origin.startswith("CompletionBuilder:"):
                    candidate_groups["completion"].append(curr_atoms)
                elif origin.startswith("InitialBuilder:"):
                    candidate_groups["initial"].append(curr_atoms)
                else:
                    ...

        return candidate_groups

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
            frames.extend(generated)
        if len(frames) != size:
            raise RuntimeError(
                f"Builder {name!r} generated {len(frames)} of {size} requested structures "
                f"after {maximum_attempts} attempts."
            )
        return frames

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
            frames = clean_seed_structures(frames)
            for atoms in frames:
                atoms.info["data"] = {"builder": name}
                atoms.info["key_value_pairs"] = dict(origin=f"InitialBuilder:{name}", extinct=0)
            starting_population.extend(frames)
        if len(starting_population) != self.init_size:
            raise RuntimeError("Failed to generate the configured initial population.")
        return starting_population

    def _prepare_current_population(
        self,
        database: GODB,
        curr_gen: int,
        builders: Mapping,
        operators: dict,
        candidate_groups: Optional[dict] = None,
    ) -> list[Atoms]:
        """Prepare current population.

        Usually, it should be the same as the initial size. However, for variable
        composition search, a large init size can be useful.

        Args:
            database: database
            curr_gen: current generation
            builders: named structure builders
            pairing: pairing
            mutations: mutations
            candidate_groups: candidate groups
        Returns:
            A list of Atoms.

        """
        assert self.population is not None
        population = self.population

        candidate_groups = candidate_groups or {}
        paired_structures = list(candidate_groups.get("paired", []))
        mutated_structures = list(candidate_groups.get("mutated", []))
        completion_structures = list(candidate_groups.get("completion", []))

        # We need adjust n_top for the variable composition search.
        num_atoms_substrate = database.get_param("num_atoms_substrate")
        assert isinstance(num_atoms_substrate, int)

        plan = database.get_generation_plan(curr_gen)
        if plan is None:
            plan = {"stage": "reproduction"}
            database.set_generation_plan(curr_gen, plan)

        if plan["stage"] == "reproduction":
            for i in range(self.gen_rep_max_try):
                if len(paired_structures) >= self.gen_rep_size:
                    break
                self._print(f"Reproduction attempt {i} ->")
                atoms = self._reproduce(
                    database,
                    curr_gen,
                    population,
                    operators,
                    num_atoms_substrate,
                )
                if atoms is not None:
                    paired_structures.append(atoms)
                    parents = " ".join([str(x) for x in atoms.info["data"]["parents"]])
                    self._print(
                        f"  confid={atoms.info['confid']:>6d} parents={parents:<14s} origin={atoms.info['key_value_pairs']['origin']:<20s} extinct={atoms.info['key_value_pairs']['extinct']:<4d}"
                    )
                else:
                    self._print(f"  reproduction failed")
            plan = {"stage": "mutation"}
            database.set_generation_plan(curr_gen, plan)

        if plan["stage"] == "mutation":
            for i in range(self.gen_mut_max_try):
                if len(mutated_structures) >= self.gen_mut_size:
                    break
                self._print(f"Mutation attempt {i} ->")
                parent = population.get_one_candidate(with_history=True)
                assert isinstance(parent, Atoms)
                atoms, desc = operators["mobile"]["mutations"].get_new_individual([parent])
                if atoms is not None:
                    database.add_unrelaxed_candidate(
                        atoms, description=desc, origin="MutationCandidateUnrelaxed", generation=curr_gen
                    )
                    mutated_structures.append(atoms)
            deficit = self.gen_size - len(paired_structures) - len(mutated_structures)
            if deficit < 0:
                raise RuntimeError("Reproduction and mutation exceeded generation.total_size.")
            plan = {"stage": "completion", "completion_sizes": self.allocate_completion_sizes(deficit)}
            database.set_generation_plan(curr_gen, plan)

        if plan["stage"] == "completion":
            self._require_builders(builders, (x["builder"] for x in plan["completion_sizes"]))
            existing = {}
            for atoms in completion_structures:
                name = atoms.info.get("data", {}).get("builder")
                existing[name] = existing.get(name, 0) + 1
            for allocation in plan["completion_sizes"]:
                name, target = allocation["builder"], allocation["size"]
                remaining = target - existing.get(name, 0)
                if remaining < 0:
                    raise RuntimeError(f"Too many persisted completion structures for builder {name!r}.")
                frames = self._generate_from_builder(
                    name, builders[name], remaining, allocation["maximum_attempts"]
                )
                for atoms in frames:
                    atoms.info.setdefault("data", {})["builder"] = name
                    database.add_unrelaxed_candidate(
                        atoms,
                        description=f"builder: {name}",
                        origin=f"CompletionBuilder:{name}",
                        generation=curr_gen,
                    )
                    completion_structures.append(atoms)
            plan = dict(plan, stage="complete")
            database.set_generation_plan(curr_gen, plan)

        current_candidates = paired_structures + mutated_structures + completion_structures
        if len(current_candidates) != self.gen_size:
            raise RuntimeError(
                f"Generation {curr_gen} contains {len(current_candidates)} candidates; expected {self.gen_size}."
            )
        return current_candidates

    def _update_generation_settings(self, mutations, pairing):
        """Update some generation-specific attributes of the operators."""
        assert self.population is not None
        candidates = self.population.get_current_population()

        # mutations
        for mut in mutations.oplist:
            if hasattr(mut, "update_scaling_volume"):
                mut.update_scaling_volume(candidates, w_adapt=0.5, n_adapt=0)
                self._print(f"{mut.__class__.__name__:<32s} scaling volume: {mut.scaling_volume:>12.4f}")

        # crossover
        if hasattr(pairing, "update_scaling_volume"):
            pairing.update_scaling_volume(candidates, w_adapt=0.5, n_adapt=0)
            self._print(f"{pairing.__class__.__name__:<32s} scaling volume: {pairing.scaling_volume:>12.4f}")

        return

    def _reproduce(
        self,
        database: GODB,
        curr_gen: int,
        population,
        operators: dict,
        num_atoms_substrate: int,
    ) -> Optional[Atoms]:
        """Reproduce a structure from the current population.

        Args:
            curr_gen: The current generation number.

        Returns:
            An atoms.

        """
        # Get pairing and mutations from operators
        pairing = operators["mobile"]["pairing"]
        mutations = operators["mobile"]["mutations"]

        custom_mutations = None
        if operators.get("custom", None) is not None:
            custom_mutations = operators["custom"]["mutations"]

        # Check if we have enough structures for pairing
        num_structures_in_population = len(population.pop)
        if not (num_structures_in_population > 0):
            raise RuntimeError(
                "Not enough structures in the current population. Some errors must have occurred before."
            )

        if num_structures_in_population >= 2:
            if pairing.allow_variable_composition:
                parents = population.get_two_candidates()
                natoms_p0, natoms_p1 = len(parents[0]), len(parents[1])
                self._print(f"  p0_natoms: {natoms_p0} p1_natoms: {natoms_p1}")
            else:
                for _ in range(100):
                    parents = population.get_two_candidates()
                    # TODO: Move this check to population?
                    if parents is not None:
                        self._print(
                            f"  compare candidates: {parents[0].info['confid']:>6d} {parents[1].info['confid']:>6d}"
                        )
                    if not is_reproduction_isolation(parents):
                        if self.substrate_dtol > 0.0:
                            is_substrate_similar = compare_two_atoms_by_substrates(
                                parents[0],
                                parents[1],
                                dtol=self.substrate_dtol,
                            )
                            dmax = parents[0].info.pop("dmax", -1.0)
                            self._print(f"    substrate consistency: {dmax=:>4.2f} ({self.substrate_dtol:>4.2f})")
                            if not is_substrate_similar:
                                continue
                        # get two candidates that are both consistent in composition and substrate
                        natoms_p0 = len(parents[0])
                        tags_dict = get_tags_per_species(parents[0])
                        identities = " ".join([k + "_" + str(len(v)) for k, v in tags_dict.items()])
                        self._print(f"  natoms: {natoms_p0} composition: {identities}")
                        break
                else:
                    self._print(
                        f"Cannot find two parents after 100 attempts from a population of {len(population.pop)}."
                    )
                    self._print(f"Get one parent and perform parthenogenesis.")
                    parent_0 = population.get_one_candidate()
                    assert parent_0 is not None
                    parents = [parent_0]
                    natoms_p0 = len(parents[0])
        else:
            # We only have one structure
            parents = [copy.deepcopy(population.pop[0])]
            natoms_p0 = len(parents[0])

        # HACK: We need adjust n_top of some operators for comptability.
        prev_substrate = None
        curr_substrate = None
        if self.substrate_dtol > 0.0:
            prev_substrate = database.get_substrate()
            substrate_indices = [i for i, tag in enumerate(parents[0].get_tags()) if tag == 0]
            curr_substrate = copy.deepcopy(parents[0][substrate_indices])

        curr_ntop = natoms_p0 - num_atoms_substrate
        if hasattr(pairing, "n_top"):
            prev_ntop = pairing.n_top
            self._print(f"  pairing  {prev_ntop =} -> {curr_ntop =}")
            pairing.n_top = curr_ntop
            assert natoms_p0 == pairing.n_top + num_atoms_substrate
        else:
            prev_ntop = curr_ntop

        if hasattr(pairing, "slab") and curr_substrate is not None:
            pairing.slab = curr_substrate

        for mutation in mutations.oplist:
            if hasattr(mutation, "n_top"):
                self._print(f"  mutation  {mutation.n_top =} -> {curr_ntop =}")
                mutation.n_top = curr_ntop
                assert natoms_p0 == mutation.n_top + num_atoms_substrate
            if hasattr(mutation, "slab") and curr_substrate is not None:
                mutation.slab = curr_substrate

        # Perform the crossover.
        a3 = None
        if len(parents) == 2:
            # This also adds key_value_pairs to a.info
            is_parthenogenesis = False
            a3, desc = pairing.get_new_individual(parents)
        else:  # Not enough structures in the current population
            # TODO: Better refactor codes here to separate
            #       amphigenesis and parthenogenesis
            is_parthenogenesis = True
            a3, desc = (
                parents[0],
                f"pairing: {parents[0].info['confid']} {parents[0].info['confid']}",
            )
            a3.info["data"] = dict(parents=[parents[0].info["confid"], parents[0].info["confid"]])
            a3.info["key_value_pairs"]["origin"] = "Parthenogenesis"

        chem_form = a3.get_chemical_formula() if a3 is not None else None
        self._print(f"  {is_parthenogenesis=}  {chem_form=}")

        # Perform mutations.
        num_mutations = len(mutations.oplist)
        if a3 is not None and num_mutations > 0:
            # Add the paired or mutated structure to the database
            a3.info["key_value_pairs"]["generation"] = curr_gen
            database.add_unrelaxed_candidate(
                a3,
                description=desc,  # here, desc is used to add "pairing": 1 to database
            )
            self._print(f"  confid= {a3.info['confid']} ")

            # mutate atoms in the mobile group
            # a3 may be changed from a valid offspring to None due to parthenogenesis
            curr_prob = self.rng.random()
            if curr_prob < self.pmut or is_parthenogenesis:
                a3_mut, mut_desc = mutations.get_new_individual([a3])
                if a3_mut is not None:
                    database.add_unrelaxed_step(a3_mut, mut_desc)
                    a3 = a3_mut
                    self._print(f"  mobile: {desc}  {mut_desc}")
                else:
                    self._print(f"  mobile: {desc}")  # Mutate failed.
                    if is_parthenogenesis:
                        self._print("  single-parent reproduction-mutation failed.")
                        a3 = None  # single-parent reproduction must mustate
            else:
                self._print(f"  mobile: {desc}")  # No mutation is applied.

            # mutate atoms in the custom group
            if custom_mutations is not None:
                curr_prob = self.rng.random()
                if curr_prob < self.pmut_custom:
                    a3_bmut, bmut_desc = custom_mutations.get_new_individual([a3])
                    if a3_bmut is not None:
                        database.add_unrelaxed_step(a3_bmut, bmut_desc)
                        a3 = a3_bmut
                        self._print(f"  custom: {bmut_desc}")
                    else:
                        ...
                else:
                    ...
            else:
                ...
        else:
            ...  # Reproduce failed.

        # restorre n_top, custom mutations should not have n_top...
        if hasattr(pairing, "n_top"):
            pairing.n_top = prev_ntop

        if hasattr(pairing, "slab") and prev_substrate is not None:
            pairing.slab = prev_substrate

        for mutation in mutations.oplist:
            if hasattr(mutation, "n_top"):
                mutation.n_top = prev_ntop
            if hasattr(mutation, "slab") and prev_substrate is not None:
                mutation.slab = prev_substrate

        return a3
