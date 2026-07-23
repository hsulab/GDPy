#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import enum
import itertools
import pathlib
import shutil
from typing import Callable, Mapping, Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.formula import Formula
from ase.io import read, write

from gdpx.compute.runtime import create_computer as convert_input_to_computer
from gdpx.compute.runtime import run_workers as run_worker
from gdpx.factory.builder import canonicalise_builder
from gdpx.factory.components import create_comparator
from gdpx.factory.computer import canonicalise_worker
from gdpx.geometry.spatial import get_bond_distance_dict
from gdpx.utils.atoms_tags import get_tags_per_species
from gdpx.utils.strconv import integers_to_string

from ..expedition import BaseExpedition
from ..persist.database import GlobalOptimisationDatabase
from ..persist.thanos import dispatch_thanos
from .utils import parse_operators, select_operator

GenerationState = enum.Enum(
    "GenerationState",
    (
        "BEG_OF_GEN",
        "MID_OF_GEN",
        "END_OF_GEN",
        "EXTINCTED",
    ),
)


def infer_unique_atomic_numbers(
    operators,
    custom_atomic_types: Optional[list[str]] = None,
    substrates: Optional[list[Atoms]] = None,
) -> list[int]:
    """Find possible elements in the simulation and build a bond-distance list."""
    type_list = []
    for op in operators:
        # TODO: wee need further unify the names here
        if hasattr(op, "particles"):
            for p in op.particles:
                type_list.extend(list(Formula(p).count().keys()))
        elif hasattr(op, "species"):
            type_list.extend(list(Formula(op.species).count().keys()))
        elif hasattr(op, "reservoir"):
            type_list.extend(list(Formula(op.reservoir["species"]).count().keys()))
        else:
            ...
    if custom_atomic_types is not None:
        type_list.extend(custom_atomic_types)
        type_list = list(set(type_list))
    if substrates is not None:
        for atoms in substrates:
            type_list = list(set(type_list + atoms.get_chemical_symbols()))
    unique_atomic_numbers = [atomic_numbers[a] for a in type_list]

    return unique_atomic_numbers


def compute_population_fitness(structures: list[Atoms], with_history=True) -> list[float]:
    """Calculates the fitness."""
    scores = [x.info["key_value_pairs"]["raw_score"] for x in structures]
    min_s = min(scores)
    max_s = max(scores)
    T = min_s - max_s

    f = [0.5 * (1.0 - np.tanh(2.0 * (s - max_s) / T - 1.0)) for s in scores]
    if with_history:
        M = [float(atoms.info["n_paired"]) for atoms in structures]
        L = [float(atoms.info["looks_like"]) for atoms in structures]
        f = [f[i] * 1.0 / np.sqrt(1.0 + M[i]) * 1.0 / np.sqrt(1.0 + L[i]) for i in range(len(f))]

    return f


class ConcurrentPopulation:
    def __init__(
        self,
        initial_size: int,
        generation_size: int,
        random_offspring_generator: dict,
        comparator: Optional[dict] = None,
        thanos: Optional[dict] = None,
        population_size: Optional[int] = None,
        database_fname: str = "mydb.db",
        print_func=print,
        debug_func=print,
    ) -> None:
        """"""
        # Name of attached databse
        if not database_fname.endswith("db"):
            raise Exception("`database_fname` must end with db.")
        self.database_fname = database_fname

        # Population sizes
        self._ini_size = initial_size
        self._gen_size = generation_size

        if population_size is not None:
            self._pop_size = population_size
        else:
            self._pop_size = self._gen_size

        if self.ini_size < self.pop_size:
            raise RuntimeError(
                f"`initial_size`({self.ini_size}) must be greater than `population_size`({self.pop_size})."
            )

        if self.pop_size < self.gen_size:
            raise RuntimeError(
                f"`population_size`({self.pop_size}) must be greater than or equal `generation_size`({self.gen_size})."
            )

        # This can be `None` as it may be lazy-initialised by builder externally.
        self.random_offspring_generator = canonicalise_builder(random_offspring_generator)

        # Comparator adds history information for atoms in the population
        if comparator is None:
            from ase.ga.standard_comparators import AtomsComparator

            self.comparator = AtomsComparator()
        else:
            name = comparator.pop("name", "interatomic_distance")
            self.comparator = create_comparator(dict(method=name, **comparator))

        # Thanos (observer/describer) extincts structures in the population
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

        # Print and debug
        self._print = print_func
        self._debug = debug_func

        return

    @property
    def ini_size(self):
        """The number of structures in the initial generation."""

        return self._ini_size

    @property
    def gen_size(self):
        """The number of structures in the following generations."""

        return self._gen_size

    @property
    def pop_size(self):
        """The number of structures in the population."""

        return self._pop_size

    def get_current_population(self, database: "GlobalOptimisationDatabase", use_extinct: bool = False) -> list[Atoms]:
        """"""
        all_relaxed_candidates = database.get_all_relaxed_candidates(use_extinct=use_extinct)
        # The candidates have already been sorted by raw_score,
        # here, we just double check it.
        all_relaxed_candidates.sort(key=lambda cand: cand.info["key_value_pairs"]["raw_score"], reverse=True)

        # We may not have enough structures for the population as some of them may look like.
        # TODO: Cache candidates?
        selected_candidates = []
        for candidate in all_relaxed_candidates:
            for s_cand in selected_candidates:
                if self.comparator.looks_like(candidate, s_cand):
                    break
            else:
                selected_candidates.append(candidate)
            num_candidates = len(selected_candidates)
            if num_candidates == self.pop_size:
                break
        else:
            ...  # Not enough candidates to select

        def count_looks_like(a, all_cand, comp):
            """Utility method for counting occurrences."""
            n = 0
            for b in all_cand:
                if a.info["confid"] == b.info["confid"]:
                    continue
                if comp.looks_like(a, b):
                    n += 1
            return n

        for s_cand in selected_candidates:
            s_cand.info["looks_like"] = count_looks_like(s_cand, selected_candidates, self.comparator)

        # TODO: Check history?
        for s_cand in selected_candidates:
            s_cand.info["n_paired"] = 0

        num_selected = len(selected_candidates)
        self._print(f"population: [{num_selected}/{self.pop_size}]")
        for i, s_cand in enumerate(selected_candidates):
            self._debug(
                f"cand{i:>4d} looks_like->{s_cand.info['looks_like']:>04d} n_paired->{s_cand.info['n_paired']:>04d}"
            )

        return selected_candidates

    def get_current_generation(
        self,
        database: "GlobalOptimisationDatabase",
        rng: np.random.Generator,
        with_history: bool = True,
        use_extinct: bool = False,
    ) -> list[Atoms]:
        """"""
        popultion = self.get_current_population(database, use_extinct=use_extinct)
        num_structures_in_population = len(popultion)

        if num_structures_in_population <= self.gen_size:
            selected_candidates = popultion
        else:
            fit = compute_population_fitness(popultion, with_history=with_history)
            fit = np.array(fit)
            weights = fit / np.sum(fit)
            cand_indices = list(range(num_structures_in_population))
            selected_indices = rng.choice(
                cand_indices, size=self.gen_size, p=weights, replace=True
            )  # TODO: allow same candidate?
            selected_candidates = [popultion[i] for i in selected_indices]

        num_selected = len(selected_candidates)
        self._print(f"generation: [{num_selected}/{self.gen_size}]")

        return selected_candidates


def run_monte_carlo_steps(
    atoms: Atoms,
    identifier: int,
    driver,
    operators,
    probabilities,
    mcsteps: int,
    rng: np.random.Generator,
) -> tuple[Atoms, list[int]]:
    """"""
    mctraj_fpath = driver.directory.parent / "mctrajs" / f"mc-{identifier:>04d}.xyz"
    energy_before = atoms.get_potential_energy()
    atoms.info["mcstep"] = 0
    write(mctraj_fpath, atoms, append=False)

    mcstates = []
    for istep in range(1, mcsteps + 1):
        op = select_operator(operators, probabilities, rng=rng)  # type: ignore
        op._print(f"  >>> mcmove.{istep:>04d}")
        atoms.calc = None  # Clear calc as exchange may break atoms arrays such as forces.
        new_atoms = op.run(atoms, rng=rng)
        if new_atoms is not None:
            ...
        else:
            ...  # try again?

        if new_atoms is not None:
            _ = driver.run(new_atoms, read_ckpt=True)
            relaxed_atoms = driver.read_trajectory()[-1]
            energy_after = relaxed_atoms.get_potential_energy()
            op._print(f"  ene {energy_before:>18.4f}  {energy_after:>18.4f}")
            success = op.metropolis(energy_before, energy_after, rng=rng)
            if success:
                atoms = relaxed_atoms
                atoms.info["mcstep"] = istep
                energy_before = energy_after
                write(mctraj_fpath, atoms, append=True)
                op._print(f"  <<< success")
                mcstates.append(0)
            else:
                # atoms should be reverted in metropolis
                op._print(f"  <<< revert")
                mcstates.append(1)

            # Remove the computation results.
            shutil.rmtree(driver.directory)
        else:
            step_state = "MCOPFAILED"
            mcstates.append(2)

    atoms.info.pop("mcstep")

    return atoms, mcstates


def evaluate_candidate(atoms: Atoms, target_property: str, chempot: Optional[dict] = None) -> None:
    """Evaluate the candidate's fitness.

    The fitness is stored in atoms.info['raw_score'].
    The candidate is better with a larger raw_score.

    The supported properties are

        1. energy (potential energy)
        2. enthalpy (potential energy plus pressure correction)
        3. formation_energy (grand canonical)
        4. reaction_energy (TODO)

    Args:
        atoms: The candidate with calculated properties.

    Returns:
        None.

    """
    assert atoms.info["key_value_pairs"].get("raw_score", None) is None, (
        "candidate already has raw_score before evaluation"
    )

    # evaluate based on target property
    target = target_property
    if target == "energy":
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()  # TODO: Make sure we have forces?
        atoms.info["key_value_pairs"]["raw_score"] = -energy
        atoms.info["key_value_pairs"]["target"] = energy
        # TODO: Check bulk structure?
    elif target == "formation_energy":
        chempot_dict = chempot
        assert chempot is not None, "`chempot` must not be None for `formation_energy`."
        identity_stats = atoms.info.get("identity_stats", None)
        assert identity_stats is not None, (
            "Fail to compute `formation_energy` as no `identity_stats` is found in atoms.info."
        )

        energy = atoms.get_potential_energy()

        formation_energy = energy - np.sum([chempot_dict[k] * v for k, v in identity_stats.items()])  # type: ignore
        atoms.info["key_value_pairs"]["raw_score"] = -formation_energy
        atoms.info["key_value_pairs"]["target"] = formation_energy
    elif target == "reaction_energy":
        ...  # TODO: ...
    else:
        raise RuntimeError(f"Unknown target {target}...")

    return


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


def canonical_candidates_from_worker_results(
    relaxed_candidates: list[Atoms],
    gen_num: int,
    use_tags: bool = False,
    property: dict = {},
    extinct_callbacks: Optional[list[Callable]] = None,
) -> list[Atoms]:
    """"""
    target_property = property.get("target", "energy")
    chempot = property.get("chempot", None)

    for candidate in relaxed_candidates:
        extra_info = dict(
            data={},
            key_value_pairs={
                "generation": gen_num,
            },
        )
        candidate.info.update(extra_info)
        # update molecular identity tags
        if use_tags:
            # The worker respects tags in atom, thus, we do not need
            # get tags from the database.
            # rows = list(self.da.c.select(f"relaxed=0,gaid={confid}"))
            # rows = sorted(
            #     [row for row in rows if row.formula], key=lambda row: row.mtime
            # )
            # if len(rows) > 0:
            #     previous_atoms = rows[-1].toatoms(
            #         add_additional_information=True
            #     )
            #     previous_tags = previous_atoms.get_tags()
            # else:
            #     raise RuntimeError(f"Cannot find tags for candidate {confid}")
            # cand.set_tags(previous_tags)
            identities = get_tags_per_species(candidate)
            identity_stats = {}
            for k, v in identities.items():
                identity_stats[k] = len(v)
            candidate.info["identity_stats"] = identity_stats
        # add raw score
        evaluate_candidate(candidate, target_property=target_property, chempot=chempot)
        # extinct if needed
        if extinct_callbacks is not None:
            extinct_candidate(candidate, extinct_callbacks=extinct_callbacks)

    return relaxed_candidates


class ConcurrentHopping(BaseExpedition):
    def __init__(
        self,
        operators: list[dict],
        num_mcmoves: int,
        mcworker: dict,
        population: dict,
        convergence: dict,
        property: dict,
        builder=None,
        use_archive: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialise ConcurrentHopping.

        Args:
            builder: Builder parameters.
            operators: Operator parameters.
            population: Population parameters.

        """
        super().__init__(*args, **kwargs)

        # Store initial parameters
        self._init_params = copy.deepcopy(
            dict(
                num_mcmoves=num_mcmoves,
                operators=operators,
                mcworker=mcworker,
                population=population,
                builder=builder,
                convergence=convergence,
                property=property,
            )
        )

        # population
        self.population = ConcurrentPopulation(**population)

        if builder is not None:
            builder = canonicalise_builder(builder)
            self.population.random_offspring_generator = builder
            self._print("Overwrite random_offspring_generator externally.")

        # Parse monte carlo settings
        self.num_mcmoves = num_mcmoves
        self.operators, self.op_probs = parse_operators(operators)
        self.mcworker = canonicalise_worker(mcworker)

        # Some convergence criteria
        self.convergence = convergence

        # The target optimised property
        self.property = property

        # Whether perform extinction after generation
        self.use_extinct = True if self.population.extinct_callbacks is not None else False

        # Whether archive results after run_worker
        self.use_archive = use_archive

        return

    def register_worker(self, worker: dict, *args, **kwargs) -> None:  # type: ignore
        """Overwrite this function as we need computer in this expedition."""
        self.worker = worker if isinstance(worker, list) else convert_input_to_computer(worker)

        return

    def run(self):
        """"""
        self._print(f"===== Concurrent Hopping =====")
        # Make sure we have everything for the expedition
        # assert isinstance(self.worker, DriverBasedWorker)

        # Try to connect to a database
        database_fpath = self.directory / self.population.database_fname
        database = GlobalOptimisationDatabase(database_fpath=database_fpath)

        # Update print and debug functions
        self.population._print = self._print
        self.population._debug = self._debug
        self._print(f"comparator: {self.population.comparator.__class__.__name__}")
        self._print("")

        for op in self.operators:
            op._print = self._print
            op._debug = self._debug
            op.indent = "  "
            for l in str(op).splitlines():
                self._print(l)
            self._print("")

        # Register minimum covalent bond distance used by operators
        # TODO: Maker a better interface?
        bond_distance_dict = {}
        if hasattr(self.population.random_offspring_generator, "get_bond_distance_dict"):
            bond_distance_dict.update(
                self.population.random_offspring_generator.get_bond_distance_dict()  # type: ignore
            )
        unique_atomic_numbers = infer_unique_atomic_numbers(
            operators=self.operators, custom_atomic_types=None, substrates=None
        )
        bond_distance_dict.update(get_bond_distance_dict(unique_atomic_numbers=unique_atomic_numbers, ratio=1.0))

        custom_pair_distance_dict = {}
        if hasattr(self.population.random_offspring_generator, "get_custom_pair_distance_dict"):
            custom_pair_distance_dict.update(
                self.population.random_offspring_generator.get_custom_pair_distance_dict()  # type: ignore
            )

        for op in self.operators:
            op.bond_distance_dict = bond_distance_dict
            op.custom_pair_distance_dict = custom_pair_distance_dict

        # Run generations
        for _ in range(1000):
            gen_num, gen_state = self.get_generation_info(database=database)
            converged = self.read_convergence(database=database, gen_num=gen_num, gen_state=gen_state)
            self._print(f"Generation info: {gen_num=}  {gen_state=}  {converged=}")
            if not converged:
                is_finished = self._irun(database=database, gen_num=gen_num, gen_state=gen_state)
                if not is_finished:
                    self._print("Wait generation to finish.")
                    break  # Wait for the step to finish.
            else:
                self.report(database)
                break  # The expedition is converged.

        return

    def _irun(
        self,
        database: GlobalOptimisationDatabase,
        gen_num: int,
        gen_state: GenerationState,
    ):
        """Run one generation."""
        # Check whether we should move on to next generation
        if gen_state == GenerationState.END_OF_GEN:
            gen_num += 1
        self._print(f"===== Generation {gen_num:>04d} =====")

        # We store all computation files in the folder below
        gen_wdir = self.directory / "tmp_folder" / f"gen{gen_num}"
        gen_wdir.mkdir(parents=True, exist_ok=True)

        # Run the initial population
        if gen_num == 0:
            # The first generation (gen-0)
            # TODO: If the initial random is failed?
            structures = self.population.random_offspring_generator.run(size=self.population.ini_size)  # type: ignore
            num_structures = len(structures)
            self._print(f"The initial population {num_structures=}.")
            for atoms in structures:
                database.add_unrelaxed_candidate(candidate=atoms)
        else:
            # TODO: How about if we are in the middle of a generation?
            # assert gen_state != GenerationState.MID_OF_GEN, "Cannot handle mid-generation yet."

            # We save all mc trajectories in a centralised folder
            (gen_wdir / "mctrajs").mkdir(parents=True, exist_ok=True)
            # Try to generate new structures
            candidates = sorted(
                self.population.get_current_generation(
                    database=database, rng=self.rng, with_history=True, use_extinct=self.use_extinct
                ),
                key=lambda a: a.info["confid"],
            )
            candidates_confids = [a.info["confid"] for a in candidates]
            self._print(f"confids {integers_to_string(candidates_confids, inp_convention='lmp')}")

            for icand, candidate in enumerate(candidates):
                self._print(f">>>>> cand{icand} confid {candidate.info['confid']}")
                self.mcworker.driver.directory = gen_wdir / f"mc_{icand}"
                atoms = copy.deepcopy(candidate)
                atoms_after_mc, mcstates = run_monte_carlo_steps(
                    atoms,
                    identifier=icand,
                    driver=self.mcworker.driver,
                    operators=self.operators,
                    probabilities=self.op_probs,
                    mcsteps=self.num_mcmoves,
                    rng=self.rng,
                )
                database.add_unrelaxed_candidate(atoms_after_mc)
                self._print(
                    f"<<<<< cand{icand} confid {candidate.info['confid']} state {''.join([str(s) for s in mcstates])}"
                )

        # Run simulations in the generation folder
        candidates_to_explore = sorted(
            database.get_all_unrelaxed_candidates(mark_as_queued=True), key=lambda a: a.info["confid"]
        )
        candidates_confids = [a.info["confid"] for a in candidates_to_explore]
        self._print(f"confids {integers_to_string(candidates_confids, inp_convention='lmp')}")

        is_finished = run_worker(candidates_to_explore, self.worker, archive=self.use_archive, directory=gen_wdir)  # type: ignore
        if is_finished:
            relaxed_candidates = read(gen_wdir / "results" / "end_frames.xyz", ":")
            explored_candidates = canonical_candidates_from_worker_results(
                relaxed_candidates,  # type: ignore
                gen_num=gen_num,
                use_tags=True,
                property=self.property,
                extinct_callbacks=self.population.extinct_callbacks,
            )
            if self.use_extinct:
                num_extincts = sum(
                    1 for candidate in explored_candidates if candidate.info["key_value_pairs"].get("extinct", 0) == 1
                )
                self._print(f"Extincted {num_extincts} candidates in generation {gen_num}.")
            # Store relaxed candidates
            for candidate in explored_candidates:
                database.add_relaxed_step(candidate)

        return is_finished

    def read_convergence(
        self,
        database: Optional[GlobalOptimisationDatabase] = None,
        gen_num: Optional[int] = None,
        gen_state: Optional[GenerationState] = None,
    ) -> bool:
        """"""
        maximum_generation_number = self.convergence.get("generation", 0)

        # We may check convergence externally, for example, by worker,
        # thus, the generation need to be determined here.
        # Otherwise, internally, we can reuse pre-determined info.
        if gen_num is None:
            if database is None:
                database_fpath = self.directory / self.population.database_fname
                database = GlobalOptimisationDatabase(database_fpath)
            gen_num, gen_state = self.get_generation_info(database=database)
        else:
            assert gen_state is not None

        if gen_num == maximum_generation_number and gen_state == GenerationState.END_OF_GEN:
            converged = True
        elif gen_num > maximum_generation_number and gen_state == GenerationState.BEG_OF_GEN:
            assert gen_num == maximum_generation_number + 1, f"{gen_num=}  {gen_state=}"
            converged = True
        elif gen_state == GenerationState.EXTINCTED:
            self._print(":( candidates are extincted...")
            converged = True
        else:
            converged = False

        return converged

    def get_generation_info(self, database: GlobalOptimisationDatabase) -> tuple[int, GenerationState]:
        """"""
        ini_size, gen_size = self.population.ini_size, self.population.gen_size

        # def get_generation_state(number_rest, number_target):
        #     """"""
        #     if number_rest == 0:
        #         gen_state = GenerationState.BEG_OF_GEN
        #     elif number_rest < number_target:
        #         gen_state = GenerationState.MID_OF_GEN
        #     elif number_rest == number_target:
        #         gen_state = GenerationState.END_OF_GEN
        #     else:
        #         raise Exception("This should not happen.")
        #
        #     return gen_state
        #
        # # Determine the stage of the generation by number of relaxed candidates
        # number_relaxed = database.get_number_of_relaxed_candidates()
        # if number_relaxed <= ini_size:  # Still in the initial generation
        #     gen_state = get_generation_state(number_relaxed, ini_size)
        #     gen_num = 0
        # else:
        #     number_finished_generations = int((number_relaxed - ini_size) / gen_size)
        #     assert number_finished_generations >= 0
        #     gen_state = get_generation_state(
        #         number_relaxed - number_finished_generations * gen_size - ini_size,
        #         gen_size,
        #     )
        #     gen_num = number_finished_generations + 1

        # Since we may use extinction, we cannot infer generation by total number of candidates already relaxed.
        def is_dir_nonempty(p: pathlib.Path) -> bool:
            """"""
            is_nonempty = False
            for f in p.rglob("*"):
                if f.is_file() and f.stat().st_size > 0:
                    is_nonempty = True
                    break

            return is_nonempty

        gen_num, gen_state = 0, GenerationState.BEG_OF_GEN
        found_state = False

        gen_wdirs = sorted((self.directory / "tmp_folder").glob("gen*"), key=lambda p: int(p.name[3:]), reverse=True)
        for gen_wdir in gen_wdirs:
            gen_num = int(gen_wdir.name[3:])
            if (gen_wdir / "results" / "end_frames.xyz").exists():
                gen_state = GenerationState.END_OF_GEN
                found_state = True
            else:
                # Have any non-empty cand folder?
                cand_wdirs = sorted(gen_wdir.glob("cand*"), key=lambda p: int(p.name[4:]))
                for cand_wdir in cand_wdirs:
                    if is_dir_nonempty(cand_wdir):
                        gen_state = GenerationState.MID_OF_GEN
                        found_state = True
                        break
            if found_state:
                break

        # Check if all structures are extincted at the end of generation
        if gen_state == GenerationState.END_OF_GEN and self.use_extinct:
            all_relaxed_candidates = database.get_all_relaxed_candidates(use_extinct=True)
            num_survived = len(all_relaxed_candidates)
            if num_survived == 0:
                gen_state = GenerationState.EXTINCTED

        return gen_num, gen_state

    def report(self, database: Optional[GlobalOptimisationDatabase] = None):
        """"""
        if database is None:
            database_fpath = self.directory / self.population.database_fname
            db = GlobalOptimisationDatabase(database_fpath)
        else:
            db = database

        results_folder = self.directory / "results"
        results_folder.mkdir(parents=True, exist_ok=True)

        all_relaxed_candidates = db.get_all_relaxed_candidates(use_extinct=False)
        write(results_folder / "all_candidates.xyz", all_relaxed_candidates)

        # Plot generations
        candidates_by_generations = {}
        for k, v in itertools.groupby(
            all_relaxed_candidates,
            key=lambda a: a.info["key_value_pairs"]["generation"],
        ):
            if k in candidates_by_generations:
                candidates_by_generations[k].extend(v)
            else:
                candidates_by_generations[k] = list(v)

        target = self.property.get("target", "energy")
        maximum_generation_number = max(candidates_by_generations.keys()) + 1

        data = []
        for i in range(maximum_generation_number):
            candidates = candidates_by_generations[i]
            properties = np.array([a.info["key_value_pairs"]["target"] for a in candidates])
            stats = dict(
                min=np.min(properties),
                max=np.max(properties),
                avg=np.mean(properties),
                std=np.std(properties),
            )
            self._print(
                f"num {properties.shape[0]:>4d} min {stats['min']:>12.4f} max {stats['max']:>12.4f} avg {stats['avg']:>12.4f} std {stats['std']:>12.4f}"
            )
            data.append([i, properties])

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        ax.set_title("Population Evolution")  # type: ignore
        for i, properties in data:
            ax.scatter([i] * len(properties), properties, alpha=0.5)  # type: ignore
        ax.set(xlabel="generation", xticks=range(maximum_generation_number), ylabel=target)  # type: ignore
        fig.savefig(results_folder / "pop.png", bbox_inches="tight")
        plt.close()

        return

    def get_workers(self):
        """Get all workers used by this expedition.

        This should always be called after the convergence is confirmed.

        Note:
            We do not have MC trajectories for now.

        """
        gen_wdirs = (self.directory / "tmp_folder").glob("gen*")
        gen_wdirs = sorted(gen_wdirs, key=lambda p: int(p.name[3:]))
        self._print(f"{gen_wdirs=}")

        workers = []
        for gen_wdir in gen_wdirs:
            gen_worker = copy.deepcopy(self.worker)
            gen_worker.directory = gen_wdir
            workers.append(gen_worker)

        return workers

    def as_dict(self) -> dict:
        """"""
        params = copy.deepcopy(self._init_params)
        params["method"] = "concurrent_hopping"
        assert self.worker is not None
        params["worker"] = self.worker.as_dict()  # type: ignore
        params["random_seed"] = self.random_seed

        return params


if __name__ == "__main__":
    ...
