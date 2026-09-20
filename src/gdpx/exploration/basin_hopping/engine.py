import copy
import itertools
import pathlib
from typing import Callable, Optional

import numpy as np
from ase import Atoms
from ase.io import read, write

from gdpx.execution.lifecycle.runtime import create_runtime_workers, execute_workers
from ..population.random import RandomStreamRegistry
from .selection import HoppingStartSelector
from ..population import Population
from ..population.config import PopulationConfig
from ..population.comparators import create_population_comparator
from gdpx.execution.factory import create_worker
from gdpx.execution.workers.worker import BaseWorker
from gdpx.structures.geometry.spatial import get_bond_distance_dict
from gdpx.utils.atoms_tags import get_tags_per_species
from gdpx.utils.strconv import integers_to_string

from ..expedition import BaseExpedition
from ..objective import is_default_objective, normalise_objective, reject_legacy_property
from ..persist.database import CANDIDATES_DATABASE_FILENAME, GlobalOptimisationDatabase
from gdpx.sampling import parse_operators
from ..generation import GenerationInfo, GenerationState, EvaluationStatus, restore_generation_random_states
from .chain import run_hopping_steps
from gdpx.sampling.geometry import infer_unique_atomic_numbers, prepare_operators

def evaluate_candidate(
    atoms: Atoms,
    objective_target: str,
    chemical_potentials: Optional[dict] = None,
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
    assert atoms.info["key_value_pairs"].get("raw_score", None) is None, (
        "candidate already has raw_score before evaluation"
    )

    # Evaluate the configured objective.
    target = objective_target
    if target == "energy":
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()  # TODO: Make sure we have forces?
        atoms.info["key_value_pairs"]["raw_score"] = -energy
        atoms.info["key_value_pairs"]["target"] = energy
        # TODO: Check bulk structure?
    elif target == "formation_energy":
        assert chemical_potentials is not None, (
            "chemical_potentials must not be None for formation_energy."
        )
        identity_stats = atoms.info.get("identity_stats", None)
        assert identity_stats is not None, (
            "Fail to compute `formation_energy` as no `identity_stats` is found in atoms.info."
        )

        energy = atoms.get_potential_energy()

        formation_energy = energy - np.sum(
            [chemical_potentials[k] * v for k, v in identity_stats.items()]
        )
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
    objective: Optional[dict] = None,
    extinct_callbacks: Optional[list[Callable]] = None,
) -> list[Atoms]:
    """"""
    objective = normalise_objective(objective, {"energy", "formation_energy"})
    objective_target = objective["target"]
    chemical_potentials = objective.get("chemical_potentials")

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
        evaluate_candidate(
            candidate,
            objective_target=objective_target,
            chemical_potentials=chemical_potentials,
        )
        # extinct if needed
        if extinct_callbacks is not None:
            extinct_candidate(candidate, extinct_callbacks=extinct_callbacks)

    return relaxed_candidates


class BasinHopping(BaseExpedition):
    def __init__(
        self,
        operators: list[dict],
        num_mcmoves: int,
        mcworker: dict,
        population: dict,
        convergence: dict,
        objective: Optional[dict] = None,
        builder=None,
        use_archive: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialise BasinHopping.

        Args:
            builder: Removed; use population.builders.
            operators: Operator parameters.
            population: Population parameters.

        """
        reject_legacy_property(kwargs)
        if builder is not None:
            raise ValueError("BH builder moved to population.builders and initial.builder_allocations.")
        super().__init__(*args, **kwargs)
        self.random_streams = RandomStreamRegistry(self.random_seed)
        self.rng = self.random_streams.get("engine")

        objective = normalise_objective(objective, {"energy", "formation_energy"})

        # Store initial parameters
        self._init_params = dict(
            num_mcmoves=num_mcmoves,
            operators=operators,
            mcworker=mcworker,
            population=population,
        )
        if not is_default_objective(objective):
            self._init_params["objective"] = copy.deepcopy(objective)
        self._init_params.update(convergence=convergence, use_archive=use_archive)

        self.population_config = PopulationConfig(population, rng=self.random_streams.get("population"))
        unsupported = {"reproduction", "mutation", "completion"} & population["generation"].keys()
        if unsupported:
            raise ValueError("BH generation does not accept GA policies: " + ", ".join(sorted(unsupported)))
        self.builders = self.population_config.initialise_builders(population, self.random_streams)
        self.generator = self.builders[self.population_config.reference_builder_name]
        comparator = create_population_comparator(
            self.population_config.comparator_config, self.population_config.periodic,
            self.random_streams.get("population/comparator"),
        )
        self.population = Population(
            self.population_config.retained_size, comparator, self.population_config.use_extinct,
        )
        self.start_selector = HoppingStartSelector(self.random_streams.get("population"))

        # Parse monte carlo settings
        self.num_mcmoves = num_mcmoves
        self.operators, self.op_probs = parse_operators(operators)
        self.mcworker = create_worker(mcworker)

        # Some convergence criteria
        self.convergence = convergence

        # The search objective
        self.objective = objective

        # Whether perform extinction after generation
        self.use_extinct = True if self.population_config.extinct_callbacks is not None else False

        # Whether archive results after run_worker
        self.use_archive = use_archive

        return

    @property
    def database_path(self) -> pathlib.Path:
        """Return the fixed candidate database path for this expedition."""
        return self.directory / CANDIDATES_DATABASE_FILENAME

    def register_worker(self, worker, *args, **kwargs) -> None:
        """Accept constructed CLI workers as well as runtime configurations."""
        if isinstance(worker, BaseWorker):
            self.worker = [worker]
        else:
            self.worker = worker if isinstance(worker, list) else create_runtime_workers(worker)

        return

    def run(self):
        """"""
        self._print(f"===== Basin Hopping =====")
        # Make sure we have everything for the expedition
        # assert isinstance(self.worker, DriverBasedWorker)

        # Try to connect to a database
        database = GlobalOptimisationDatabase(database_fpath=self.database_path)
        self._configure_generations(database)

        # Update print and debug functions
        self.population_config._print = self._print
        self.population_config._debug = self._debug
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
        if hasattr(self.generator, "get_bond_distance_dict"):
            try:
                bond_distance_dict.update(self.generator.get_bond_distance_dict())
            except NotImplementedError:
                # File/direct builders inherit the unsupported base method.
                # Actual candidate elements are added before each hopping chain.
                pass
        unique_atomic_numbers = infer_unique_atomic_numbers(
            operators=self.operators, custom_atomic_types=None, substrates=None
        )
        bond_distance_dict.update(get_bond_distance_dict(unique_atomic_numbers=unique_atomic_numbers, ratio=1.0))

        custom_pair_distance_dict = {}
        if hasattr(self.generator, "get_custom_pair_distance_dict"):
            custom_pair_distance_dict.update(
                self.generator.get_custom_pair_distance_dict()  # type: ignore
            )

        prepare_operators(self.operators, unique_atomic_numbers, bond_distance_dict, custom_pair_distance_dict)

        # Run generations
        for _ in range(1000):
            gen_info = database.get_generation_info()
            converged = self.read_convergence(gen_info=gen_info)
            self._print(f"Generation info: {gen_info} converged={converged}")
            if not converged:
                status = self._irun(database, gen_info)
                if status is EvaluationStatus.PENDING:
                    self._print("Wait generation to finish.")
                    break
            else:
                self.report(database)
                break  # The expedition is converged.

        return

    def _configure_generations(self, database):
        database.configure_generations(self.population_config.init_size, self.population_config.gen_size,
                                       self.population_config.use_extinct)
        # Old BH inputs lacked generation tags. Infer only from committed
        # results; unresolved legacy inputs cannot safely be assigned/replayed.
        for row in list(database.connection.select(relaxed=0)):
            if "generation" not in row and row.formula:
                relaxed = list(database.connection.select(confid=row.confid, relaxed=1))
                if not relaxed:
                    raise ValueError("Legacy BH pending inputs have no generation checkpoint; start a new run.")
                database.connection.update(row.id, generation=relaxed[-1].generation)

    def _prepare_generation(self, database, gen_num, gen_wdir):
        restore_generation_random_states(database, gen_num, self.random_streams)
        plan = database.get_generation_plan(gen_num)
        candidates = database.generation_candidates(gen_num)
        target = self.population_config.init_size if gen_num == 0 else self.population_config.gen_size
        if plan is None:
            if candidates:
                if len(candidates) != target:
                    raise ValueError("Legacy partial BH generation has no production checkpoint; start a new run.")
                plan = dict(stage="complete", random_states=self.random_streams.snapshot())
            else:
                plan = dict(stage="initial" if gen_num == 0 else "hopping")
                if gen_num > 0:
                    self.population.refresh(database)
                    starts = sorted(self.start_selector.select(self.population, target), key=lambda a: a.info["confid"])
                    if not starts:
                        raise RuntimeError("No eligible parents for BH generation.")
                    plan["parents"] = [a.info["confid"] for a in starts]
                plan["random_states"] = self.random_streams.snapshot()
            database.set_generation_plan(gen_num, plan)
        self.random_streams.restore(plan["random_states"])
        if plan["stage"] == "initial":
            for allocation in self.population_config.initial_builder_allocations:
                name = allocation["builder"]
                count = sum(a.info["data"].get("builder") == name for a in candidates)
                frames = self.population_config._generate_from_builder(
                    name, self.builders[name], allocation["size"] - count, allocation["maximum_attempts"])
                # Persist one allocation and its post-generation RNG atomically.
                with database.connection:
                    for atoms in self.population_config.clean_initial_structures(frames, name):
                        database.add_unrelaxed_candidate(atoms, generation=gen_num)
                        candidates.append(atoms)
                    plan["random_states"] = self.random_streams.snapshot()
                    database.set_generation_plan(gen_num, plan)
        elif plan["stage"] == "hopping":
            for index in range(len(candidates), len(plan["parents"])):
                parent = database.get_one_candidate_by_confid(plan["parents"][index])
                atoms = parent.copy()
                atoms.info = copy.deepcopy(parent.info)
                atoms.calc = parent.calc
                self.mcworker.driver.directory = gen_wdir / f"mc_{index}"
                endpoint, states = run_hopping_steps(
                    atoms, index, self.mcworker.driver, self.operators, self.op_probs,
                    self.num_mcmoves, self.rng, checkpoint_directory=gen_wdir / "chains" / f"chain-{index:04d}")
                endpoint.info["key_value_pairs"] = {}
                endpoint.info["data"] = {"parents": [plan["parents"][index]], "chain": index}
                with database.connection:
                    database.add_unrelaxed_candidate(endpoint, generation=gen_num)
                    candidates.append(endpoint)
                    plan["random_states"] = self.random_streams.snapshot()
                    database.set_generation_plan(gen_num, plan)
        if len(candidates) != target:
            raise RuntimeError(f"Generation {gen_num} has {len(candidates)} inputs; expected {target}.")
        plan["stage"] = "complete"
        database.set_generation_plan(gen_num, plan)
        return candidates

    def _irun(self, database: GlobalOptimisationDatabase, gen_info: GenerationInfo) -> EvaluationStatus:
        gen_num = gen_info.num
        gen_wdir = self.directory / "tmp_folder" / f"gen{gen_num}"
        gen_wdir.mkdir(parents=True, exist_ok=True)
        candidates = self._prepare_generation(database, gen_num, gen_wdir)
        # Include queued and already ingested candidates to preserve batch IDs
        # when the worker resumes. It owns idempotent submission/retrieval.
        finished = execute_workers(candidates, self.worker, archive=self.use_archive, directory=gen_wdir)
        if not finished:
            return EvaluationStatus.PENDING
        committed = set(database.get_generation_info(gen_num).relaxed_confids)
        expected = {a.info["confid"] for a in candidates}
        results = read(gen_wdir / "results" / "end_frames.xyz", ":")
        for candidate in results:
            confid = candidate.info["confid"]
            if confid not in expected:
                raise RuntimeError("Worker returned a candidate outside the current generation.")
            if confid in committed:
                continue
            canonical_candidates_from_worker_results(
                [candidate], gen_num=gen_num, use_tags=True, objective=self.objective,
                extinct_callbacks=self.population_config.extinct_callbacks)
            database.add_relaxed_step(candidate)
            committed.add(confid)
        if database.get_generation_info(gen_num).state is GenerationState.END_OF_GEN:
            return EvaluationStatus.FINISHED
        return EvaluationStatus.PENDING

    def read_convergence(self, database=None, gen_info=None) -> bool:
        if gen_info is None:
            database = database or GlobalOptimisationDatabase(self.database_path)
            self._configure_generations(database)
            gen_info = database.get_generation_info()
        return gen_info.converged(self.convergence.get("generation", 0))

    def report(self, database: Optional[GlobalOptimisationDatabase] = None):
        """"""
        if database is None:
            db = GlobalOptimisationDatabase(self.database_path)
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

        target = self.objective["target"]
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
            prototypes = self.worker if isinstance(self.worker, list) else [self.worker]
            if prototypes and isinstance(prototypes[0], list):
                entries = [(worker, gen_wdir / f"chainstep.{i:02d}") for i, worker in enumerate(prototypes[0])]
            else:
                entries = [(worker, gen_wdir if len(prototypes) == 1 else gen_wdir / f"w{i}")
                           for i, worker in enumerate(prototypes)]
            for prototype, directory in entries:
                # Runtime specifications are immutable and cannot be deep-copied.
                gen_worker = create_worker(prototype.as_dict(), directory=directory)
                workers.append(gen_worker)

        return workers

    def as_dict(self) -> dict:
        """"""
        recipe = {key: copy.deepcopy(value) for key, value in self._init_params.items() if key != "population"}
        recipe["population"] = self.population_config.serialise(self._init_params["population"])
        recipe = dict(random_seed=self.random_seed, **recipe)
        assert self.worker is not None

        def serialize_workers(workers):
            if isinstance(workers, list):
                return [serialize_workers(worker) for worker in workers]
            return workers.as_dict()

        runtime = serialize_workers(self.worker)
        if isinstance(runtime, list) and len(runtime) == 1 and isinstance(runtime[0], dict):
            runtime = runtime[0]
        return {
            "method": "basin_hopping",
            "recipe": recipe,
            "runtime": runtime,
        }
