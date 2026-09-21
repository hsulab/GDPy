import copy
import itertools
import pathlib
from collections.abc import Mapping
from typing import Callable, Optional

import numpy as np
from ase import Atoms
from ase.io import write

from gdpx.execution.lifecycle.runtime import create_runtime_workers
from ..population.random import RandomStreamRegistry
from .selection import HoppingStartSelector
from .output import GenerationReporter, bh_logging, report_setup
from ..sampling.logging import MoveLog
from ..population import Population
from ..population.config import PopulationConfig
from ..population.comparators import create_population_comparator
from gdpx.execution.factory import create_worker
from gdpx.execution.workers.worker import BaseWorker
from gdpx.structures.geometry.spatial import get_bond_distance_dict
from gdpx.utils.atoms_tags import get_tags_per_species

from ..expedition import BaseExpedition
from ..objective import is_default_objective, normalise_objective, reject_legacy_property
from ..persist.database import CANDIDATES_DATABASE_FILENAME, GlobalOptimisationDatabase
from ..sampling import parse_operators
from ..generation import GenerationInfo, GenerationState, EvaluationStatus, restore_generation_random_states
from .chain import evaluate_batch, run_hopping_rounds, finalize_checkpoints
from ..sampling.geometry import infer_unique_atomic_numbers, prepare_operators

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
        population: dict,
        convergence: Optional[dict] = None,
        objective: Optional[dict] = None,
        builder=None,
        use_archive: bool = True,
        selection: Optional[dict] = None,
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
        if selection is not None and not isinstance(selection, Mapping):
            raise TypeError("BH selection must be a mapping.")
        selection = {"replace": False, **dict(selection or {})}
        if selection.keys() - {"replace"}:
            raise ValueError("Unknown BH selection settings: " + ", ".join(sorted(selection.keys() - {"replace"})))
        if not isinstance(selection["replace"], bool):
            raise TypeError("BH selection.replace must be a boolean.")
        if "mcworker" in kwargs:
            raise ValueError("BH mcworker was removed; move calculation settings into top-level runtime.")
        if isinstance(num_mcmoves, bool) or not isinstance(num_mcmoves, int) or num_mcmoves < 0:
            raise ValueError("BH num_mcmoves must be a non-negative integer.")
        if convergence is not None and not isinstance(convergence, Mapping):
            raise TypeError("BH convergence must be a mapping.")
        convergence = {"generation": 1, **copy.deepcopy(dict(convergence or {}))}
        generation = convergence["generation"]
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("BH convergence.generation must be a non-negative integer.")
        if builder is not None:
            raise ValueError("BH builder moved to population.builders and initial.builder_allocations.")
        super().__init__(*args, **kwargs)
        self.random_streams = RandomStreamRegistry(self.random_seed)
        self.rng = self.random_streams.get("engine")

        objective = normalise_objective(objective, {"energy", "formation_energy"})

        # Store initial parameters
        self._init_params = dict(
            num_mcmoves=num_mcmoves,
            selection=selection,
            operators=operators,
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
        self.start_selector = HoppingStartSelector(self.random_streams.get("population"), **selection)

        # Parse monte carlo settings
        self.num_mcmoves = num_mcmoves
        self.operators, self.op_probs = parse_operators(operators)

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
        if not isinstance(worker, BaseWorker) and not hasattr(worker, "run"):
            workers = worker if isinstance(worker, list) else create_runtime_workers(worker)
            if len(workers) != 1 or isinstance(workers[0], list):
                raise ValueError("BH requires one calculation runtime.")
            worker = workers[0]
        self.worker = worker

        return

    def run(self):
        with bh_logging():
            return self._run()

    def _run(self):
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
            self._debug(f"operator: {op.as_dict()}")
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

        report_setup(self.operators, self.op_probs)

        # Run generations
        for _ in range(1000):
            gen_info = database.get_generation_info()
            converged = self.read_convergence(gen_info=gen_info)
            if not converged:
                gen_num = gen_info.num
                reporter = GenerationReporter(
                    database, self.directory / "tmp_folder" / f"gen{gen_num}" / "rounds",
                    gen_num, self.convergence["generation"], self.population_config.gen_size,
                    self.num_mcmoves, self.population_config.init_size, self.objective["target"],
                    resumed=database.get_generation_plan(gen_num) is not None)
                self._generation_reporter = reporter
                try:
                    with reporter.as_parent():
                        status = self._irun(database, gen_info)
                    plan = database.get_generation_plan(gen_num) or {}
                    if status is EvaluationStatus.PENDING:
                        detail = (f"waiting for round {reporter.step + 1}/{self.num_mcmoves} evaluations"
                                  if gen_num else "waiting for initialization evaluations")
                        reporter.finish("waiting", detail)
                    else:
                        extinct = plan.get("termination_reason") == "extinct"
                        if not gen_num:
                            extinct = not any(not row.get("extinct", 0) for row in
                                              database.connection.select(relaxed=1, generation=0,
                                                                         columns=['id', 'key_value_pairs']))
                        reporter.finish("extinct" if extinct else "complete")
                except BaseException as error:
                    # Output must not obscure the original calculation failure.
                    try:
                        reporter.finish("failed", f"{type(error).__name__}: {error}")
                    except Exception:
                        pass
                    raise
                finally:
                    self._generation_reporter = None
                if status is EvaluationStatus.PENDING:
                    break
            else:
                self.report(database)
                break  # The expedition is converged.

        return

    def _configure_generations(self, database):
        database.configure_generations(self.population_config.init_size, self.population_config.gen_size,
                                       self.population_config.use_extinct)
        for rounds in (self.directory / "tmp_folder").glob("gen*/rounds"):
            generation = int(rounds.parent.name[3:])
            plan = database.get_generation_plan(generation)
            if plan and plan.get("stage") == "complete" and plan.get("round_version") == 5:
                finalize_checkpoints(rounds)
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
        if gen_num > 0 and (
            (plan is not None and plan.get("round_version") != 5)
            or (plan is None and (candidates or (gen_wdir / "chains").exists()))
        ):
            raise ValueError("Incompatible serial BH checkpoint or older round checkpoint; start a new run.")
        target = self.population_config.init_size if gen_num == 0 else self.population_config.gen_size
        if plan is None:
            if candidates:
                if gen_num == 0 and len(candidates) != target:
                    raise ValueError("Legacy partial BH generation has no production checkpoint; start a new run.")
                plan = dict(stage="complete", random_states=self.random_streams.snapshot())
            else:
                plan = dict(stage="initial" if gen_num == 0 else "hopping", round_version=5)
                if gen_num > 0:
                    self.population.refresh(database)
                    plan["population"] = [a.info["confid"] for a in self.population.candidates]
                    starts = sorted(self.start_selector.select(self.population, target), key=lambda a: a.info["confid"])
                    if not starts:
                        raise RuntimeError("No eligible parents for BH generation.")
                    plan["parents"] = [a.info["confid"] for a in starts]
                    plan["expected_confids"] = []
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
            starts = [database.get_one_candidate_by_confid(confid) for confid in plan["parents"]]
            for start, confid in zip(starts, plan["parents"]):
                start.info["confid"] = confid

            def record_trial(step, chain, trial, accepted, parent, segment, start_parent):
                canonical_candidates_from_worker_results(
                    [trial], gen_num=gen_num, use_tags=True, objective=self.objective,
                    extinct_callbacks=self.population_config.extinct_callbacks)
                extinct = bool(trial.info["key_value_pairs"].get("extinct", 0))
                trial.info["data"].update(
                    parents=[parent], chain=chain, round=step, accepted=bool(accepted),
                    start_parent=start_parent, segment=segment,
                    outcome="extinct" if accepted and extinct else ("accepted" if accepted else "rejected"))
                database.add_evaluated_candidate(trial, f"bh:{gen_num}:{chain}:{step}")
                return extinct

            def restart_chains(terminated, step):
                # Recovery from the previous checkpoint must not see later
                # round results already ingested before the interruption.
                self.population.refresh(database, history=database.get_all_relaxed_candidates(
                    use_extinct=self.population.use_extinct, through=(gen_num, step)))
                selected = self.start_selector.select(self.population, len(terminated))
                replacements = []
                for candidate in selected:
                    confid = candidate.info["confid"]
                    replacement = database.get_one_candidate_by_confid(confid)
                    replacement.info["confid"] = confid
                    replacements.append(replacement)
                return replacements

            with MoveLog(gen_wdir / 'mcmoves.log',
                         gen_num, self.operators, self.op_probs) as move_logger:
                outcome = run_hopping_rounds(
                    starts, self.worker, self.operators, self.op_probs, self.num_mcmoves,
                    self.rng, gen_wdir / "rounds", archive=self.use_archive, record_trial=record_trial,
                    restart_chains=restart_chains, random_streams=self.random_streams,
                    store_history=False, move_logger=move_logger,
                    on_progress=(getattr(self, "_generation_reporter", None).progress
                                 if getattr(self, "_generation_reporter", None) is not None else None))
            if outcome.status is EvaluationStatus.PENDING:
                return None
            if outcome.extinct:
                plan["termination_reason"] = "extinct"
            plan["expected_confids"] = sorted(
                row.confid for row in database.connection.select(relaxed=1, generation=gen_num))
            plan["random_states"] = self.random_streams.snapshot()
        if gen_num == 0 and len(candidates) != target:
            raise RuntimeError(f"Generation {gen_num} has {len(candidates)} inputs; expected {target}.")
        plan["stage"] = "complete"
        database.set_generation_plan(gen_num, plan)
        if gen_num > 0:
            finalize_checkpoints(gen_wdir / "rounds")
        return candidates

    def _irun(self, database: GlobalOptimisationDatabase, gen_info: GenerationInfo) -> EvaluationStatus:
        gen_num = gen_info.num
        gen_wdir = self.directory / "tmp_folder" / f"gen{gen_num}"
        gen_wdir.mkdir(parents=True, exist_ok=True)
        candidates = self._prepare_generation(database, gen_num, gen_wdir)
        if candidates is None:
            return EvaluationStatus.PENDING
        # Every hopping evaluation was already persisted by the round coordinator.
        results = evaluate_batch(candidates, self.worker, gen_wdir, self.use_archive) if gen_num == 0 else []
        if results is None:
            return EvaluationStatus.PENDING
        committed = set(database.get_generation_info(gen_num).relaxed_confids)
        expected = {a.info["confid"] for a in candidates}
        for candidate in results:
            confid = candidate.info["confid"]
            if confid not in expected:
                raise RuntimeError("Worker returned a candidate outside the current generation.")
            if confid in committed:
                continue
            provenance = candidate.info.get("data", {})
            canonical_candidates_from_worker_results(
                [candidate], gen_num=gen_num, use_tags=True, objective=self.objective,
                extinct_callbacks=self.population_config.extinct_callbacks)
            candidate.info["data"].update(provenance)
            database.add_relaxed_step(candidate)
            committed.add(confid)
        if database.get_generation_info(gen_num).state in (GenerationState.END_OF_GEN, GenerationState.EXTINCTED):
            return EvaluationStatus.FINISHED
        return EvaluationStatus.PENDING

    def read_convergence(self, database=None, gen_info=None) -> bool:
        if gen_info is None:
            database = database or GlobalOptimisationDatabase(self.database_path)
            self._configure_generations(database)
            gen_info = database.get_generation_info()
        return gen_info.converged(self.convergence["generation"])

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

        from .lineage import plot_lineage
        plot_lineage(db.connection, self.directory)

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
        maximum_generation_number = max(max(candidates_by_generations.keys(), default=0),
                                        db.get_generation_number() - 1) + 1

        data = []
        for i in range(maximum_generation_number):
            candidates = candidates_by_generations.get(i, [])
            if not candidates:
                self._print(f"generation {i}: no evaluated candidates")
                continue
            properties = np.array([a.info["key_value_pairs"]["target"] for a in candidates])
            stats = dict(
                min=np.min(properties),
                max=np.max(properties),
                avg=np.mean(properties),
                std=np.std(properties),
            )
            self._debug(
                f"generation {i}: {properties.shape[0]} candidates; {target} statistics: {stats}"
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

        """
        root = self.directory / "tmp_folder"
        directories = [root / "gen0"]
        directories.extend(sorted(root.glob("gen*/evaluations/round-*")))
        workers = [create_worker(self.worker.as_dict(), directory=directory)
                   for directory in directories if directory.exists()]

        return workers

    def as_dict(self) -> dict:
        """"""
        recipe = {key: copy.deepcopy(value) for key, value in self._init_params.items() if key != "population"}
        recipe["population"] = self.population_config.serialise(self._init_params["population"])
        recipe = dict(random_seed=self.random_seed, **recipe)
        assert self.worker is not None

        return {
            "method": "basin_hopping",
            "recipe": recipe,
            "runtime": self.worker.as_dict(),
        }
