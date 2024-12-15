#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import enum
import itertools
import shutil
from typing import List, Optional, Tuple

import ase.db
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.formula import Formula
from ase.io import read, write

from gdpx.geometry.spatial import get_bond_distance_dict
from gdpx.cli.compute import run_worker, convert_input_to_computer

from .. import DriverBasedWorker, get_tags_per_species, registers
from ..expedition import AbstractExpedition, canonicalise_builder, canonicalise_worker
from .operators import parse_operators, select_operator

GenerationState = enum.Enum(
    "GenerationState", ("BEG_OF_GEN", "MID_OF_GEN", "END_OF_GEN")
)


def infer_unique_atomic_numbers(
    operators,
    custom_atomic_types: Optional[List[str]] = None,
    substrates: Optional[List[Atoms]] = None,
) -> List[int]:
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


def compute_population_fitness(
    structures: List[Atoms], with_history=True
) -> List[float]:
    """Calculates the fitness."""
    scores = [x.info["key_value_pairs"]["raw_score"] for x in structures]
    min_s = min(scores)
    max_s = max(scores)
    T = min_s - max_s

    f = [0.5 * (1.0 - np.tanh(2.0 * (s - max_s) / T - 1.0)) for s in scores]
    if with_history:
        M = [float(atoms.info["n_paired"]) for atoms in structures]
        L = [float(atoms.info["looks_like"]) for atoms in structures]
        f = [
            f[i] * 1.0 / np.sqrt(1.0 + M[i]) * 1.0 / np.sqrt(1.0 + L[i])
            for i in range(len(f))
        ]

    return f


class ConcurrentPopulation:

    def __init__(
        self,
        initial_size: int,
        generation_size: int,
        random_offspring_generator: dict,
        comparator: Optional[dict] = None,
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
        self.random_offspring_generator = canonicalise_builder(
            random_offspring_generator
        )

        # Comparator adds history information for atoms in the population
        if comparator is None:
            from ase.ga.standard_comparators import AtomsComparator

            self.comparator = AtomsComparator()
        else:
            name = comparator.pop("name", "interatomic_distance")
            self.comparator = registers.create("comparator", name, **comparator)

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

    def get_current_population(
        self, database: "GlobalOptimisationDatabase"
    ) -> List[Atoms]:
        """"""
        all_relaxed_candidates = database.get_all_relaxed_candidates(use_extinct=False)
        # The candidates have already been sorted by raw_score,
        # here, we just double check it.
        all_relaxed_candidates.sort(
            key=lambda cand: cand.info["key_value_pairs"]["raw_score"], reverse=True
        )

        # We may not have enough structures for the population
        # as some of them may look like.
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
            s_cand.info["looks_like"] = count_looks_like(
                s_cand, selected_candidates, self.comparator
            )

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
    ) -> List[Atoms]:
        """"""
        popultion = self.get_current_population(database)
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


class GlobalOptimisationDatabase:

    def __init__(self, database_fpath) -> None:
        """"""
        self.connection = ase.db.connect(database_fpath)

        return

    def add_unrelaxed_candidate(self, candidate: Atoms, **kwargs):
        """"""
        confid = self.connection.write(
            candidate, relaxed=0, queued=0, extinct=0, **kwargs
        )
        self.connection.update(confid, confid=confid)
        candidate.info["confid"] = confid

        return

    def add_relaxed_step(self, atoms: Atoms) -> None:
        """"""
        assert "raw_score" in atoms.info["key_value_pairs"]

        confid = atoms.info["confid"]
        rows = list(self.connection.select(confid=confid, relaxed=0))
        assert len(rows) == 1

        relax_id = self.connection.write(
            atoms,
            relaxed=1,
            confid=confid,
            key_value_pairs=atoms.info["key_value_pairs"],
            data=atoms.info["data"],
        )
        atoms.info["relax_id"] = relax_id

        return

    def get_one_candidate_by_confid(
        self, confid: int, add_info: bool = True, mark_as_queued: bool = False
    ) -> Atoms:
        """"""
        images = list(self.connection.select(confid=confid))
        images.sort(key=lambda x: x.mtime)

        # TODO: if there is no images?
        candidate = self.connection.get_atoms(
            images[-1].id, add_additional_information=add_info
        )
        if mark_as_queued:
            self.connection.update(id=images[-1].id, queued=1)

        return candidate

    def get_all_relaxed_candidates(self, use_extinct: bool = False):
        """"""
        if use_extinct:
            rows = self.connection.select("relaxed=1,extinct=0", sort="-raw_score")
        else:
            rows = self.connection.select("relaxed=1", sort="-raw_score")

        candidates = []
        for row in rows:
            candidate = self.connection.get_atoms(
                id=row.id, add_additional_information=True
            )
            candidate.info["confid"] = row.confid
            candidates.append(candidate)

        return candidates

    def get_number_of_relaxed_candidates(self):
        """"""
        confids = self._get_all_relaxed_confids()

        return len(confids)

    def _get_all_relaxed_confids(self):
        """"""
        relaxed_confids = {row.confid for row in self.connection.select(relaxed=1)}

        confids = [confid for confid in relaxed_confids]

        return confids

    def get_all_unrelaxed_candidates(self, mark_as_queued: bool = False) -> List[Atoms]:
        """"""
        confids = self._get_all_unrelaxed_confids()

        candidates = []
        for confid in confids:
            candidate = self.get_one_candidate_by_confid(
                confid, mark_as_queued=mark_as_queued
            )
            candidate.info["confid"] = confid
            if "data" not in candidate.info:
                candidate.info["data"] = {}
            candidates.append(candidate)

        return candidates

    def _get_all_unrelaxed_confids(self):
        """"""
        relaxed_confids = {row.confid for row in self.connection.select(relaxed=1)}
        unrelaxed_confids = {row.confid for row in self.connection.select(relaxed=0)}
        queued_confids = {row.confid for row in self.connection.select(queued=1)}

        confids = [
            confid
            for confid in unrelaxed_confids
            if (confid not in relaxed_confids and confid not in queued_confids)
        ]

        return confids


def run_monte_carlo_steps(
    atoms: Atoms,
    identifier: int,
    driver,
    operators,
    probabilities,
    mcsteps: int,
    rng: np.random.Generator,
) -> Atoms:
    """"""
    mctraj_fpath = (
        driver.directory.parent / "mctrajs" / f"mc-{identifier:>04d}.xyz"
    )
    energy_before = atoms.get_potential_energy()
    atoms.info["mcstep"] = 0
    write(mctraj_fpath, atoms, append=False)
    for istep in range(1, mcsteps+1):
        op = select_operator(operators, probabilities, rng=rng)  # type: ignore
        op._print(f"----- MCSTEP.{istep:>04d} -----")
        new_atoms = op.run(atoms, rng=rng)
        if new_atoms is not None:
            ...
        else:
            ...  # try again?

        if new_atoms is not None:
            _ = driver.run(new_atoms, read_ckpt=True)
            relaxed_atoms = driver.read_trajectory()[-1]
            energy_after = relaxed_atoms.get_potential_energy()
            op._print(f"{energy_before=}  {energy_after=}")
            success = op.metropolis(energy_before, energy_after, rng=rng)
            op._print(f"mcstep.{istep:>04d} {success=}")
            if success:
                atoms = relaxed_atoms
                atoms.info["mcstep"] = istep
                energy_before = energy_after
                write(mctraj_fpath, atoms, append=True)
            else:
                ...
            # Remove the computation results.
            shutil.rmtree(driver.directory)
        else:
            step_state = "MCOPFAILED"
    atoms.info.pop("mcstep")

    return atoms


def evaluate_candidate(
    atoms: Atoms, target_property: str, chempot: Optional[dict] = None
) -> None:
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
    assert (
        atoms.info["key_value_pairs"].get("raw_score", None) is None
    ), "candidate already has raw_score before evaluation"

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
        assert (
            identity_stats is not None
        ), "Fail to compute `formation_energy` as no `identity_stats` is found in atoms.info."

        energy = atoms.get_potential_energy()

        formation_energy = energy - np.sum(
            [chempot_dict[k] * v for k, v in identity_stats.items()]  # type: ignore
        )
        atoms.info["key_value_pairs"]["raw_score"] = -formation_energy
        atoms.info["key_value_pairs"]["target"] = formation_energy
    elif target == "reaction_energy":
        ...  # TODO: ...
    else:
        raise RuntimeError(f"Unknown target {target}...")

    return


def canonical_candidates_from_worker_results(
    relaxed_candidates: List[Atoms],
    gen_num: int,
    extinct: int = 0,
    use_tags: bool = False,
    property: dict = {},
):
    """"""
    target_property = property.get("target", "energy")
    chempot = property.get("chempot", None)

    for candidate in relaxed_candidates:
        extra_info = dict(
            data={}, key_value_pairs={"generation": gen_num, "extinct": extinct}
        )
        candidate.info.update(extra_info)
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

    return relaxed_candidates


class ConcurrentHopping(AbstractExpedition):

    def __init__(
        self,
        monte_carlo,
        population: dict,
        convergence: dict,
        property: dict,
        builder=None,
        worker=None,
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
                monte_carlo=monte_carlo,
                population=population,
                builder=builder,
                worker=worker,
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

        # Parse operators
        self.mcsteps = monte_carlo.get("mcsteps")
        operators = monte_carlo.get("operators")
        self.operators, self.op_probs = parse_operators(operators)
        mcworker = monte_carlo.get("mcworker")
        self.mcworker = canonicalise_worker(mcworker)

        # Some convergence criteria
        self.convergence = convergence

        # The target optimised property
        self.property = property

        # Worker should be lazy initialised before run
        self.worker = convert_input_to_computer(worker)

        return

    def register_worker(self, worker: dict, *args, **kwargs) -> None:  # type: ignore
        """Overwrite this function as we need computer in this expedition."""

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
        self._print(f"comparator: {self.population.comparator}")

        for op in self.operators:
            op._print = self._print
            op._debug = self._debug

        # Register minimum covalent bond distance used by operators
        # TODO: Maker a better interface?
        bond_distance_dict = {}
        if hasattr(
            self.population.random_offspring_generator, "get_bond_distance_dict"
        ):
            bond_distance_dict.update(
                self.population.random_offspring_generator.get_bond_distance_dict()  # type: ignore
            )
        unique_atomic_numbers = infer_unique_atomic_numbers(
            operators=self.operators, custom_atomic_types=None, substrates=None
        )
        bond_distance_dict.update(
            get_bond_distance_dict(
                unique_atomic_numbers=unique_atomic_numbers, ratio=1.0
            )
        )
        for op in self.operators:
            op.bond_distance_dict = bond_distance_dict

        # Run generations
        for _ in range(1000):
            gen_num, gen_state = self.get_generation_info(database=database)
            converged = self.read_convergence(
                database=database, gen_num=gen_num, gen_state=gen_state
            )
            if not converged:
                is_finished = self._irun(
                    database=database, gen_num=gen_num, gen_state=gen_state
                )
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
            # We save all mc trajectories in a centralised folder
            (gen_wdir / "mctrajs").mkdir(parents=True, exist_ok=True)
            # Try to generate new structures
            candidates = self.population.get_current_generation(
                database=database, rng=self.rng, with_history=True
            )
            candidates_confids = [a.info["confid"] for a in candidates]
            self._print(f"{candidates_confids=}")

            for icand, candidate in enumerate(candidates):
                self._print(f">>>>> cand{icand} confid {candidate.info['confid']}")
                self.mcworker.driver.directory = gen_wdir / f"mc_{icand}"
                atoms = copy.deepcopy(candidate)
                atoms_after_mc = run_monte_carlo_steps(
                    atoms,
                    identifier=icand,
                    driver=self.mcworker.driver,
                    operators=self.operators,
                    probabilities=self.op_probs,
                    mcsteps=self.mcsteps,
                    rng=self.rng,
                )
                database.add_unrelaxed_candidate(atoms_after_mc)

        # Run simulations in the generation folder
        candidates_to_explore = database.get_all_unrelaxed_candidates(
            mark_as_queued=True
        )
        candidates_confids = [a.info["confid"] for a in candidates_to_explore]
        self._print(f"{candidates_confids=}")

        is_finished = run_worker(candidates_to_explore, self.worker, directory=gen_wdir)  # type: ignore
        if is_finished:
            relaxed_candidates = read(gen_wdir/"results"/"end_frames.xyz", ":")
            explored_candidates = canonical_candidates_from_worker_results(
                relaxed_candidates,  # type: ignore
                gen_num=gen_num,
                extinct=0,
                use_tags=True,
                property=self.property,
            )
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
        if (
            gen_num == maximum_generation_number
            and gen_state == GenerationState.END_OF_GEN
        ):
            converged = True
        elif (
            gen_num > maximum_generation_number
            and gen_state == GenerationState.BEG_OF_GEN
        ):
            assert gen_num == maximum_generation_number + 1, f"{gen_num=}  {gen_state=}"
            converged = True
        else:
            converged = False

        return converged

    def get_generation_info(
        self, database: GlobalOptimisationDatabase
    ) -> Tuple[int, GenerationState]:
        """"""
        ini_size, gen_size = self.population.ini_size, self.population.gen_size

        def get_generation_state(number_rest, number_target):
            """"""
            if number_rest == 0:
                gen_state = GenerationState.BEG_OF_GEN
            elif number_rest < number_target:
                gen_state = GenerationState.MID_OF_GEN
            elif number_rest == number_target:
                gen_state = GenerationState.END_OF_GEN
            else:
                raise Exception("This should not happen.")

            return gen_state

        number_relaxed = database.get_number_of_relaxed_candidates()
        if number_relaxed <= ini_size:  # Still in the initial generation
            gen_state = get_generation_state(number_relaxed, ini_size)
            gen_num = 0
        else:
            number_finished_generations = int((number_relaxed - ini_size) / gen_size)
            assert number_finished_generations >= 0
            gen_state = get_generation_state(
                number_relaxed - number_finished_generations * gen_size - ini_size,
                gen_size,
            )
            gen_num = number_finished_generations + 1

        return gen_num, gen_state

    def report(self, database: Optional[GlobalOptimisationDatabase] = None):
        """"""
        if database is None:
            database_fpath = self.directory / self.population.database_fname
            database = GlobalOptimisationDatabase(database_fpath)

        results_folder = self.directory / "results"
        results_folder.mkdir(parents=True, exist_ok=True)

        all_relaxed_candidates = database.get_all_relaxed_candidates(use_extinct=False)
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
            properties = np.array(
                [a.info["key_value_pairs"]["target"] for a in candidates]
            )
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
        """"""

        raise NotImplementedError()

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
