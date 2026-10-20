"""Population search using independent MD cooling trajectories and quenched endpoints."""
import json
import os

import numpy as np

from ase.io import read, write
from ase.io.jsonio import encode

from gdpx.execution.factory import create_worker
from gdpx.execution.fingerprint import payload_digest
from ..population.exploration import PopulationBasedExploration
from ..basin_hopping.engine import canonical_candidates_from_worker_results
from ..basin_hopping.selection import HoppingStartSelector
from ..persist.database import GlobalOptimisationDatabase
from gdpx.execution.checkpoint import write_json
from ..objective import normalise_objective


class PopulationAnnealing(PopulationBasedExploration):
    """Retain diverse minima across repeated parallel cooling cycles."""

    name = "simulated_annealing"

    def __init__(self, population, strategy, convergence=None, objective=None,
                 use_archive=False, **kwargs):
        convergence = dict(convergence or {"generation": 1})
        generation = convergence.get("generation")
        if (set(convergence) != {"generation"} or isinstance(generation, bool)
                or not isinstance(generation, int) or generation < 0):
            raise ValueError("SA convergence requires a nonnegative integer generation.")
        if {"annealing", "minimisation"}.intersection(strategy):
            raise ValueError("Move annealing/minimisation to the native SA runtime setup list.")
        super().__init__(population, strategy, **kwargs)
        self.convergence = convergence
        self.objective = normalise_objective(objective, {"energy", "formation_energy"})
        self.use_archive = use_archive
        self._init_params = dict(population=population, strategy=strategy,
                                 convergence=convergence, objective=self.objective,
                                 use_archive=use_archive)
        self.selector = HoppingStartSelector(self.random_streams.get("population"))

    def register_worker(self, worker, *args, **kwargs):
        if isinstance(worker, list) and len(worker) != 1:
            raise ValueError("Population SA requires exactly one runtime.")
        candidate = worker[0] if isinstance(worker, list) and len(worker) == 1 else worker
        if len(candidate._drivers) != 1:
            raise ValueError("Population SA requires exactly one runtime.")
        config = candidate.runtime.config
        if (config.executor.provider, config.executor.method) != ("native", "simulated_annealing"):
            raise ValueError("Population SA requires a native simulated_annealing runtime.")
        if candidate.driver.convergence["steps"] != 1:
            raise ValueError("Population SA requires exactly one cooling/minimisation pass per generation.")
        super().register_worker(worker, *args, **kwargs)

    def _minimum_runtime(self):
        runtime = self.worker.driver.children[1].config.to_dict()
        runtime["dispatch"] = self.worker.runtime.config.dispatch.to_dict()
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
        runtime = self.worker.as_dict()
        md = runtime["executor"]["parameters"]["setup"][0]["parameters"]
        if {"setup", "output", "stop"}.intersection(md):
            velocities = md.setdefault("setup", {}).setdefault("velocities", {})
            velocities.update(seed=None, initialize="always")
        else:
            md.update(velocity_seed=None, ignore_atoms_velocities=True)
        return create_worker(runtime, directory=self.directory / f"gen{generation}" / "sim")

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
            worker = (self._cooling(generation, frames, plan["seed"]) if generation
                      else create_worker(self._minimum_runtime(), directory=root / "min"))
            seeds = np.random.default_rng(plan["seed"]).integers(0, 2**31 - 1, len(frames)).tolist()
            worker.run(frames, rng_states=seeds)
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
                workers.append(self._cooling(generation, read(root / "inputs.xyz", ":"), plan["seed"]))
            if (root / "min").exists():
                workers.append(create_worker(self._minimum_runtime(), directory=root / "min"))
        return workers
