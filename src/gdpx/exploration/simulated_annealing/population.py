"""Population search using independent MD cooling trajectories and quenched endpoints."""
import copy
import json
import os
import math

from ase.io import read, write
from ase.io.jsonio import encode

from gdpx.execution.factory import create_worker
from gdpx.execution.fingerprint import payload_digest
from gdpx.providers.configuration import resolve_executor_parameters
from ..population.exploration import PopulationBasedExploration
from ..basin_hopping.engine import canonical_candidates_from_worker_results
from ..basin_hopping.selection import HoppingStartSelector
from ..persist.database import GlobalOptimisationDatabase
from gdpx.execution.checkpoint import write_json
from ..objective import normalise_objective
from .simulated_annealing import SimulatedAnnealing


class PopulationAnnealing(PopulationBasedExploration):
    """Retain diverse minima across repeated parallel cooling cycles."""

    name = "simulated_annealing"

    def __init__(self, population, strategy, convergence=None, objective=None,
                 use_archive=False, **kwargs):
        # The cooling runner validates temperatures without constructing a worker.
        SimulatedAnnealing(None, annealing=strategy.get("annealing"), random_seed=0)
        convergence = dict(convergence or {"generation": 1})
        generation = convergence.get("generation")
        if (set(convergence) != {"generation"} or isinstance(generation, bool)
                or not isinstance(generation, int) or generation < 0):
            raise ValueError("SA convergence requires a nonnegative integer generation.")
        minimisation = strategy.get("minimisation")
        if not isinstance(minimisation, dict):
            raise ValueError("SA requires strategy.minimisation executor parameters.")
        params = resolve_executor_parameters(minimisation, "min")
        fmax = params.get("fmax")
        if (isinstance(params.get("steps"), bool) or not isinstance(params.get("steps"), int)
                or params["steps"] <= 0 or isinstance(fmax, bool)
                or not isinstance(fmax, (int, float)) or not math.isfinite(fmax) or fmax <= 0):
            raise ValueError("SA minimisation requires positive steps and fmax.")
        super().__init__(population, strategy, **kwargs)
        self.convergence = convergence
        self.objective = normalise_objective(objective, {"energy", "formation_energy"})
        self.use_archive = use_archive
        self._init_params = dict(population=population, strategy=strategy,
                                 convergence=convergence, objective=self.objective,
                                 use_archive=use_archive)
        self.selector = HoppingStartSelector(self.random_streams.get("population"))

    def register_worker(self, worker, *args, **kwargs):
        # Reuse the cooling runner's provider/runtime validation.
        validator = SimulatedAnnealing(None, annealing=self.strategy_config["annealing"], random_seed=0)
        validator.register_worker(worker)
        super().register_worker(worker, *args, **kwargs)
        create_worker(self._minimum_runtime())

    def _minimum_runtime(self):
        runtime = self.worker.as_dict()
        runtime["executor"]["method"] = "min"
        runtime["executor"]["parameters"] = copy.deepcopy(self.strategy_config["minimisation"])
        runtime["dispatch"]["worker"] = "batch"
        return runtime

    def as_dict(self):
        return self.serialise_search(self._init_params)

    def _checkpoint(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "population.json"
        signature = payload_digest(json.loads(encode(self.as_dict())))
        if path.exists():
            state = json.loads(path.read_text())
            if state.get("version") != 1 or state["signature"] != signature:
                raise ValueError("SA configuration changed; use a new directory.")
        else:
            if (self.directory / "candidates.db").exists():
                raise ValueError("Missing population checkpoint; use a new directory.")
            state = dict(version=1, signature=signature, completed=-1, plans={},
                         random_states=self.random_streams.snapshot())
            write_json(path, state)
        self.random_streams.restore(state["random_states"])
        return state

    def _save(self, state):
        state["random_states"] = self.random_streams.snapshot()
        write_json(self.directory / "population.json", state)

    def _cooling(self, generation, frames, seed):
        cooling = SimulatedAnnealing(
            {"method": "direct", "frames": frames}, annealing=self.strategy_config["annealing"],
            directory=self.directory / f"gen{generation}" / "sim", random_seed=seed)
        runtime = self.worker.as_dict()
        params = resolve_executor_parameters(runtime["executor"]["parameters"], "md")
        params["velocity_seed"] = None
        runtime["executor"]["parameters"] = params
        cooling.register_worker(create_worker(runtime))
        return cooling

    def run(self):
        state = self._checkpoint()
        database = GlobalOptimisationDatabase(self.directory / "candidates.db")
        for generation in range(state["completed"] + 1, self.convergence["generation"] + 1):
            key = str(generation)
            root = self.directory / f"gen{generation}"
            root.mkdir(parents=True, exist_ok=True)
            if key not in state["plans"]:
                parents = []
                if generation:
                    self.population.refresh(database)
                    starts = self.selector.select(self.population, self.population_config.gen_size)
                    if not starts:
                        raise RuntimeError("No eligible SA parents remain.")
                    parents = [a.info["confid"] for a in starts]
                state["plans"][key] = dict(parents=parents, seed=int(self.rng.integers(0, 2**31 - 1)),
                                           input_rng=self.random_streams.snapshot())
                self._save(state)
            plan = state["plans"][key]
            inputs = root / "inputs.xyz"
            if not inputs.exists():
                self.random_streams.restore(plan["input_rng"])
                frames = ([database.get_one_candidate_by_confid(p) for p in plan["parents"]]
                          if generation else self.population_config._prepare_initial_population(self.builders))
                for frame in frames:
                    frame.calc = None
                    frame.info = {}
                    if "momenta" in frame.arrays:
                        del frame.arrays["momenta"]
                # Persist the post-builder stream before publishing input structures.
                plan["after_input_rng"] = self.random_streams.snapshot()
                self._save(state)
                write(root / "inputs.tmp.xyz", frames, format="extxyz")
                os.replace(root / "inputs.tmp.xyz", inputs)
            self.random_streams.restore(plan["after_input_rng"])
            frames = read(inputs, ":")
            if generation:
                cooling = self._cooling(generation, frames, plan["seed"])
                cooling.run()
                if not cooling.read_convergence():
                    return
                frames = read(cooling.directory / "final.xyz", ":")
            worker = create_worker(self._minimum_runtime(), directory=root / "min")
            worker.run(frames, rng_states=[plan["seed"]] * len(frames))
            worker.inspect(resubmit=True)
            if worker.get_number_of_running_jobs():
                return
            results = worker.retrieve(include_retrieved=True, use_archive=self.use_archive)
            by_name = {traj[-1].info["wdir"]: traj[-1] for traj in results if traj}
            if len(results) != len(frames) or set(by_name) != {f"cand{i}" for i in range(len(frames))}:
                raise RuntimeError("Missing or duplicate SA minimisation results.")
            for index in range(len(frames)):
                atoms = by_name[f"cand{index}"]
                canonical_candidates_from_worker_results(
                    [atoms], gen_num=generation, objective=self.objective,
                    extinct_callbacks=self.population_config.extinct_callbacks)
                atoms.info["data"].update(parents=[plan["parents"][index]] if generation else [],
                                          cycle=generation, replica=index)
                database.add_evaluated_candidate(atoms, f"sa:{generation}:{index}")
            self.population.refresh(database)
            plan["retained"] = [a.info["confid"] for a in self.population.candidates]
            state["completed"] = generation
            self._save(state)
            self._print(f"SA generation {generation}: evaluated {len(frames)}, retained {len(plan['retained'])}")
        self.population.refresh(database)
        self.report(database)

    def report(self, database=None):
        """Rebuild final artefacts from persisted evaluations, without running MD."""
        from .results import write_results

        if database is None:
            database = GlobalOptimisationDatabase(self.directory / "candidates.db")
        self.population.refresh(database)
        write_results(
            self.directory / "results",
            database.get_all_relaxed_candidates(use_extinct=False),
            list(self.population.candidates),
            self.objective["target"],
        )

    def read_convergence(self):
        path = self.directory / "population.json"
        return path.exists() and json.loads(path.read_text())["completed"] >= self.convergence["generation"]

    def get_workers(self):
        state = json.loads((self.directory / "population.json").read_text())
        workers = []
        for key, plan in state["plans"].items():
            generation = int(key)
            root = self.directory / f"gen{generation}"
            if generation and (root / "inputs.xyz").exists():
                workers.extend(self._cooling(generation, read(root / "inputs.xyz", ":"), plan["seed"]).get_workers())
            if (root / "min").exists():
                workers.append(create_worker(self._minimum_runtime(), directory=root / "min"))
        return workers
