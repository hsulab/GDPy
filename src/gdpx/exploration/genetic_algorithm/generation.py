import copy
from typing import Callable, Mapping, Optional

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from gdpx.exploration.persist.database import GlobalOptimisationDatabase as GODB
from gdpx.utils.atoms_tags import get_tags_per_species
from gdpx.utils.profiler import CustomTimer

from ..population.config import PopulationConfig

#: Retained keys in key_value_pairs when get_atoms from the database.
RETAINED_KEYS: list[str] = ["extinct", "origin"]


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


class GeneticGenerationManager:
    """Produce GA generations using shared configuration, population, and a selector."""

    _print = print

    _debug = print

    @staticmethod
    def validate_parameters(params):
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
            raise ValueError("Population name must be `constant` or `variable`.")

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
        substrate_params = params.get("substrate", dict(distance_tolerance=-1.0))
        if "dtol" in substrate_params:
            raise ValueError("Legacy GA substrate key 'dtol' is not supported; use 'distance_tolerance'.")
        return name

    def __init__(self, params: dict, config: PopulationConfig, population, selector, rng):
        self.name = self.validate_parameters(params)
        self.config = config
        self.population = population
        self.selector = selector
        self.rng = rng
        gen_params = params["generation"]
        substrate_params = params.get("substrate", {})
        # Get number of structures from different origins in one generation
        reproduction_params = gen_params.get("reproduction", {})
        mutation_params = gen_params.get("mutation", {})
        completion_params = gen_params.get("completion", {})
        generation_sections = (reproduction_params, mutation_params, completion_params)
        if not all(isinstance(section, Mapping) for section in generation_sections):
            raise ValueError(
                "generation.reproduction, generation.mutation, and generation.completion must be mappings."
            )
        self.gen_rep_size = self.config._nonnegative_integer(reproduction_params.get("size", 0), "reproduction.size")
        self.gen_mut_size = self.config._nonnegative_integer(mutation_params.get("size", 0), "mutation.size")
        if self.gen_rep_size + self.gen_mut_size > self.config.gen_size:
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
        self.substrate_dtol = substrate_params.get("distance_tolerance", -1.0)  # Ang

        return


    def _attempts(self, params: Mapping, size: int, section: str) -> int:
        attempts = params.get("maximum_attempts", size * self.config.MAX_ATTEMPTS_MULTIPLIER)
        return self.config._nonnegative_integer(attempts, f"generation.{section}.maximum_attempts")


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
                maximum_attempts = self.config._nonnegative_integer(
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
                    else item_size * self.config.MAX_ATTEMPTS_MULTIPLIER
                ),
            )
            for item, item_size in zip(self.completion_builder_proportions, allocated)
        ]

    def update_population(self, database: GODB) -> None:
        """Refresh shared membership and GA-only parent-selection history."""
        self.population.refresh(database)
        self.selector.refresh(self.population, database)

    def _extinct_candidate(self, atoms: Atoms) -> None:
        """Extinct the candidate by given callback.

        Args:
            atoms: The candidate to be evaluated.

        """
        if self.config.use_extinct and self.config.extinct_callbacks is not None:
            extinct_candidate(atoms, self.config.extinct_callbacks)

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


    def _prepare_current_population(
        self,
        database: GODB,
        curr_gen: int,
        builders: Mapping,
        operators: dict,
        candidate_groups: Optional[dict] = None,
        random_state_getter=None,
        random_state_restorer=None,
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
            plan = {
                "stage": "reproduction",
                "reproduction_attempts": 0,
                "mutation_attempts": 0,
            }
            if random_state_getter is not None:
                plan["random_states"] = random_state_getter()
            database.set_generation_plan(curr_gen, plan)
        elif random_state_restorer is not None and "random_states" in plan:
            random_state_restorer(plan["random_states"])

        def checkpoint():
            if random_state_getter is not None:
                plan["random_states"] = random_state_getter()
            database.set_generation_plan(curr_gen, plan)

        if plan["stage"] == "reproduction":
            first_attempt = int(plan.get("reproduction_attempts", 0))
            for i in range(first_attempt, self.gen_rep_max_try):
                if len(paired_structures) >= self.gen_rep_size:
                    break
                self._print(f"Reproduction attempt {i} ->")
                with database.connection:
                    atoms = self._reproduce(
                        database,
                        curr_gen,
                        population,
                        operators,
                        num_atoms_substrate,
                    )
                    if atoms is not None:
                        self.config.validate_candidate(atoms, "reproduction")
                        paired_structures.append(atoms)
                        parents = " ".join([str(x) for x in atoms.info["data"]["parents"]])
                        self._print(
                            f"  confid={atoms.info['confid']:>6d} parents={parents:<14s} origin={atoms.info['key_value_pairs']['origin']:<20s} extinct={atoms.info['key_value_pairs']['extinct']:<4d}"
                        )
                    else:
                        self._print(f"  reproduction failed")
                    plan["reproduction_attempts"] = i + 1
                    checkpoint()
            plan["stage"] = "mutation"
            checkpoint()

        if plan["stage"] == "mutation":
            first_attempt = int(plan.get("mutation_attempts", 0))
            for i in range(first_attempt, self.gen_mut_max_try):
                if len(mutated_structures) >= self.gen_mut_size:
                    break
                self._print(f"Mutation attempt {i} ->")
                with database.connection:
                    parent = self.selector.select_one(population, with_history=True)
                    assert isinstance(parent, Atoms)
                    parent = parent.copy()
                    parent.info = copy.deepcopy(parent.info)
                    atoms, desc = operators["mobile"]["mutations"].get_new_individual([parent])
                    if atoms is not None:
                        self.config.validate_candidate(atoms, "mutation")
                        database.add_unrelaxed_candidate(
                            atoms, description=desc, origin="MutationCandidateUnrelaxed", generation=curr_gen
                        )
                        mutated_structures.append(atoms)
                    plan["mutation_attempts"] = i + 1
                    checkpoint()
            deficit = self.config.gen_size - len(paired_structures) - len(mutated_structures)
            if deficit < 0:
                raise RuntimeError("Reproduction and mutation exceeded generation.total_size.")
            plan["stage"] = "completion"
            plan["completion_sizes"] = self.allocate_completion_sizes(deficit)
            checkpoint()

        if plan["stage"] == "completion":
            self.config._require_builders(builders, (x["builder"] for x in plan["completion_sizes"]))
            existing = {}
            for atoms in completion_structures:
                name = atoms.info.get("data", {}).get("builder")
                existing[name] = existing.get(name, 0) + 1
            for allocation in plan["completion_sizes"]:
                name, target = allocation["builder"], allocation["size"]
                remaining = target - existing.get(name, 0)
                if remaining < 0:
                    raise RuntimeError(f"Too many persisted completion structures for builder {name!r}.")
                frames = self.config._generate_from_builder(
                    name, builders[name], remaining, allocation["maximum_attempts"]
                )
                with database.connection:
                    for atoms in frames:
                        atoms.info.setdefault("data", {})["builder"] = name
                        database.add_unrelaxed_candidate(
                            atoms,
                            description=f"builder: {name}",
                            origin=f"CompletionBuilder:{name}",
                            generation=curr_gen,
                        )
                        completion_structures.append(atoms)
                        checkpoint()
            plan["stage"] = "complete"
            checkpoint()

        current_candidates = paired_structures + mutated_structures + completion_structures
        if len(current_candidates) != self.config.gen_size:
            raise RuntimeError(
                f"Generation {curr_gen} contains {len(current_candidates)} candidates; expected {self.config.gen_size}."
            )
        return current_candidates

    def _update_generation_settings(self, mutations, pairing):
        """Update some generation-specific attributes of the operators."""
        candidates = self.population.candidates

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
        num_structures_in_population = len(population.candidates)
        if not (num_structures_in_population > 0):
            raise RuntimeError(
                "Not enough structures in the current population. Some errors must have occurred before."
            )

        if num_structures_in_population >= 2:
            if pairing.allow_variable_composition:
                parents = self.selector.select_pair(population)
                if parents is None:
                    return None
                natoms_p0, natoms_p1 = len(parents[0]), len(parents[1])
                self._print(f"  p0_natoms: {natoms_p0} p1_natoms: {natoms_p1}")
            else:
                for _ in range(100):
                    parents = self.selector.select_pair(population)
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
                            self._print(f"    substrate consistency: {is_substrate_similar}")
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
                        f"Cannot find two parents after 100 attempts from a population of {len(population.candidates)}."
                    )
                    self._print(f"Get one parent and perform parthenogenesis.")
                    parent_0 = self.selector.select_one(population)
                    assert parent_0 is not None
                    parents = [parent_0]
                    natoms_p0 = len(parents[0])
        else:
            # We only have one structure
            parents = [population.candidates[0]]
            natoms_p0 = len(parents[0])

        parents = [parent.copy() for parent in parents]
        for parent in parents:
            parent.info = copy.deepcopy(parent.info)

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
            self.config.validate_candidate(a3, "crossover")
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
                    self.config.validate_candidate(a3_mut, "reproduction mutation")
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
                        self.config.validate_candidate(a3_bmut, "custom mutation")
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
