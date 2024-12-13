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
from ase.io import write

from gdpx.geometry.spatial import get_bond_distance_dict

from .. import DriverBasedWorker
from ..expedition import AbstractExpedition, canonicalise_builder
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
        population_size: Optional[int] = None,
        mcsteps: int = 5,
        database_fname: str = "mydb.db",
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
            self._pop_size = self._gen_size
        else:
            self._pop_size = self._gen_size

        # The number of MC steps
        self._mcsteps = mcsteps

        # This can be `None` as it may be lazy-initialised by builder externally.
        self.random_offspring_generator = canonicalise_builder(
            random_offspring_generator
        )

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

    @property
    def mcsteps(self):
        """The number of MC steps."""

        return self._mcsteps

    def get_current_population(self, database: "GlobalOptimisationDatabase"):
        """"""
        all_relaxed_candidates = database.get_all_relaxed_candidates(use_extinct=False)
        # The candidates have already been sorted by raw_score,
        # here, we just double check it.
        all_relaxed_candidates.sort(
            key=lambda cand: cand.info["key_value_pairs"]["raw_score"], reverse=True
        )

        # TODO: Cache candidates?
        selected_candidates = []
        for candidate in all_relaxed_candidates:
            # TODO: Use comparator?
            selected_candidates.append(candidate)
            num_candidates = len(selected_candidates)
            if num_candidates == self.pop_size:
                break
        else:
            ...  # Not enough candidates to select

        # TODO: Check history?

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
    driver,
    operators,
    probabilities,
    mcsteps: int,
    rng: np.random.Generator,
) -> Atoms:
    """"""
    mctraj_fpath = (
        driver.directory.parent / "mctrajs" / f"mc-{driver.directory.name[4:][:-3]}.xyz"
    )
    # TODO: check tags?
    energy_before = atoms.get_potential_energy()
    write(mctraj_fpath, atoms, append=False)
    for istep in range(mcsteps):
        op = select_operator(operators, probabilities, rng=rng)  # type: ignore
        op._print(f"----- MCSTEP.{istep:>04d} -----")
        new_atoms = op.run(atoms, rng=rng)
        if new_atoms is not None:
            ...
        else:
            ...  # try again?

        if new_atoms is not None:
            # TODO: use tags?
            _ = driver.run(new_atoms, read_ckpt=True)
            relaxed_atoms = driver.read_trajectory()[-1]
            energy_after = relaxed_atoms.get_potential_energy()
            op._print(f"{energy_before=}  {energy_after=}")
            success = op.metropolis(energy_before, energy_after, rng=rng)
            op._print(f"mcstep.{istep:>04d} {success=}")
            if success:
                atoms = relaxed_atoms
                energy_before = energy_after
                write(mctraj_fpath, atoms, append=True)
            else:
                ...
            # Remove the computation results.
            shutil.rmtree(driver.directory)
        else:
            step_state = "MCOPFAILED"

    return atoms


def canonical_candidates_from_worker_results(
    results: list, gen_num: int, extinct: int = 0, use_tags: bool = False
):
    """"""
    relaxed_candidates = [t[-1] for t in results]
    for candidate in relaxed_candidates:
        extra_info = dict(
            data={}, key_value_pairs={"generation": gen_num, "extinct": extinct}
        )
        candidate.info.update(extra_info)
        if use_tags:
            # TODO: add molecular information?
            ...
        # add raw score TODO: add more targets?
        energy = candidate.get_potential_energy()
        candidate.info["key_value_pairs"]["raw_score"] = -energy

    return relaxed_candidates


class ConcurrentHopping(AbstractExpedition):

    def __init__(
        self,
        operators,
        population: dict,
        convergence: dict,
        builder=None,
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
        self._init_params = dict(
            operators=operators,
            population=population,
            builder=builder,
        )

        # population
        self.population = ConcurrentPopulation(**population)

        if builder is not None:
            builder = canonicalise_builder(builder)
            self.population.random_offspring_generator = builder
            self._print("Overwrite random_offspring_generator externally.")

        # Parse operators
        self.operators, self.op_probs = parse_operators(operators)

        # Some convergence criteria
        self.convergence = convergence

        # Worker should be lazy initialised before run
        self.worker = None

        return

    def run(self):
        """"""
        # Make sure we have everything for the expedition
        assert self.worker is not None

        # Try to connect to a database
        database_fpath = self.directory / self.population.database_fname
        database = GlobalOptimisationDatabase(database_fpath=database_fpath)

        # Update print and debug functions
        for op in self.operators:
            op._print = self._print
            op._debug = self._debug

        # Register minimum covalent bond distance used by operators
        # TODO: Maker a better interface?
        unique_atomic_numbers = infer_unique_atomic_numbers(
            operators=self.operators, custom_atomic_types=None, substrates=None
        )
        for op in self.operators:
            op.blmin = get_bond_distance_dict(
                unique_atomic_numbers=unique_atomic_numbers, ratio=op.covalent_min
            )

        # Run generations
        for _ in range(1000):
            gen_num, gen_state = self.get_generation_info(database=database)
            converged = self.read_convergence(
                database=database, gen_num=gen_num, gen_state=gen_state
            )
            if not converged:
                finished = self._irun(
                    database=database, gen_num=gen_num, gen_state=gen_state
                )
                if not finished:
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

        assert isinstance(self.worker, DriverBasedWorker)

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
            candidates = self.population.get_current_population(database=database)
            candidates_confids = [a.info["confid"] for a in candidates]
            self._print(f"{candidates_confids=}")

            for icand, candidate in enumerate(candidates):
                self._print(f">>>>> cand{icand} confid {candidate.info['confid']}")
                self.worker.driver.directory = gen_wdir / f"mc_{icand}"
                atoms = copy.deepcopy(candidate)
                atoms_after_mc = run_monte_carlo_steps(
                    atoms,
                    driver=self.worker.driver,
                    operators=self.operators,
                    probabilities=self.op_probs,
                    mcsteps=self.population.mcsteps,
                    rng=self.rng,
                )
                database.add_unrelaxed_candidate(atoms_after_mc)

        # Run simulations in the generation folder
        candidates_to_explore = database.get_all_unrelaxed_candidates(
            mark_as_queued=True
        )
        candidates_confids = [a.info["confid"] for a in candidates_to_explore]
        self._print(f"{candidates_confids=}")

        self.worker.directory = gen_wdir
        self.worker.run(candidates_to_explore)

        # Try to retrieve results
        finished = False
        self.worker.inspect(resubmit=True)
        if self.worker.get_number_of_running_jobs() == 0:
            results = self.worker.retrieve(use_archive=False)  # TODO: use_archive?
            explored_candidates = canonical_candidates_from_worker_results(
                results, gen_num=gen_num, extinct=0, use_tags=False
            )
            for candidate in explored_candidates:
                database.add_relaxed_step(candidate)
            finished = True
        else:
            ...

        return finished

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
            self._print(f"{gen_num=}  {gen_state=}  in convergence.")
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

        target = "energy"
        maximum_generation_number = max(candidates_by_generations.keys()) + 1

        data = []
        for i in range(maximum_generation_number):
            candidates = candidates_by_generations[i]
            properties = np.array([a.get_potential_energy() for a in candidates])
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
