"""Population search with MD cooling and minimisation of selected snapshots."""
import copy
import math
import json
import os
from collections.abc import Mapping

import numpy as np

from ase.io import read, write
from ase.io.jsonio import encode

from gdpx.analysis.selectors import create_selector
from gdpx.data.array import AtomsNDArray

from gdpx.providers.configuration import RuntimeConfig, child_executor, resolve_executor_parameters
from gdpx.execution.factory import create_worker
from gdpx.execution.fingerprint import payload_digest
from ..population.exploration import PopulationBasedExploration
from ..basin_hopping.engine import canonical_candidates_from_worker_results
from ..basin_hopping.selection import HoppingStartSelector
from ..persist.database import GlobalOptimisationDatabase
from gdpx.execution.checkpoint import write_json
from ..objective import normalise_objective


class PopulationAnnealing(PopulationBasedExploration):
    """Retain diverse minima across MD cooling generations."""

    name = "simulated_annealing"

    def __init__(self, population, strategy, convergence=None, objective=None,
                 use_archive=False, **kwargs):
        temperatures = strategy.get("temperatures")
        if not isinstance(temperatures, list) or any(
                not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(t, bool) or not isinstance(t, (int, float))
                       or not math.isfinite(t) or t <= 0 for t in pair)
                or pair[0] <= pair[1]
                for pair in temperatures):
            raise ValueError(
                "SA strategy.temperatures must list [start, end] cooling ranges, "
                "one per generation, with finite temperatures start > end > 0.")
        self.temperatures = [list(map(float, pair)) for pair in temperatures]
        convergence = dict(convergence or {"generation": len(temperatures)})
        generation = convergence.get("generation")
        if (set(convergence) != {"generation"} or isinstance(generation, bool)
                or not isinstance(generation, int) or generation < 0):
            raise ValueError("SA convergence requires a nonnegative integer generation.")
        if generation != len(temperatures):
            raise ValueError("SA requires one temperature range per generation (excluding initial generation 0).")
        definition = strategy.get("minimisation")
        if not isinstance(definition, Mapping) or "runtime" not in definition:
            raise ValueError("SA strategy.minimisation requires runtime and an optional selector.")
        if set(definition) - {"runtime", "selector"}:
            raise ValueError("SA minimisation accepts only runtime and selector.")
        selector_config = copy.deepcopy(definition.get("selector", {"method": "locate", "indices": "-1"}))
        create_selector(selector_config)  # Validate before any calculations are submitted.
        if not isinstance(definition["runtime"], Mapping):
            raise ValueError("SA minimisation.runtime must be a runtime mapping.")
        minimum_runtime = copy.deepcopy(dict(definition["runtime"]))
        minimisation = child_executor(minimum_runtime.get("executor"))
        if "scheduler" in minimisation:
            raise ValueError("Move the minimisation scheduler to minimisation.runtime.scheduler.")
        minimum_runtime["executor"] = minimisation
        if minimisation["method"] != "min":
            raise ValueError("SA strategy.minimisation requires method: min.")
        params = resolve_executor_parameters(minimisation.get("parameters", {}), "min")
        if (isinstance(params.get("steps"), bool) or not isinstance(params.get("steps"), int)
                or params["steps"] <= 0 or isinstance(params.get("fmax"), bool)
                or not isinstance(params.get("fmax"), (int, float))
                or not math.isfinite(params["fmax"]) or params["fmax"] <= 0):
            raise ValueError("SA minimisation requires positive stop.steps and stop.fmax.")
        strategy = copy.deepcopy(strategy)
        strategy["minimisation"] = {"runtime": minimum_runtime, "selector": selector_config}
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
        if config.executor.method != "md":
            raise ValueError("Population SA requires an MD runtime; SA itself remains an exploration.")
        params = resolve_executor_parameters(config.executor.parameters, "md")
        if params.get("ensemble") not in {"nvt", "npt"}:
            raise ValueError("SA requires thermostatted NVT or NPT dynamics.")
        if (isinstance(params.get("steps"), bool) or not isinstance(params.get("steps"), int)
                or params["steps"] <= 0):
            raise ValueError("SA MD stop.steps must be positive.")
        if params.get("tend") is not None:
            raise ValueError("Remove MD tend; strategy.temperatures defines the cooling range for each generation.")
        if config.modifiers or config.dispatch.share_workdir:
            raise ValueError("SA requires an unmodified potential and separate candidate directories.")
        super().register_worker(worker, *args, **kwargs)
        create_worker(self._minimum_runtime())

    def _minimum_runtime(self):
        definition = copy.deepcopy(self.strategy_config["minimisation"]["runtime"])
        md_runtime = self.worker.as_dict()
        definition.setdefault("potential", md_runtime["potential"])
        dispatch = definition.get("dispatch", {})
        if not isinstance(dispatch, Mapping):
            raise ValueError("SA minimisation.runtime.dispatch must be a mapping.")
        definition["dispatch"] = {**md_runtime["dispatch"], "worker": "batch", **dispatch}
        runtime = RuntimeConfig.from_mapping(definition)
        if runtime.modifiers or runtime.dispatch.share_workdir:
            raise ValueError("SA minimisation requires an unmodified potential and separate candidate directories.")
        if runtime.dispatch.worker != "batch":
            raise ValueError("SA minimisation requires dispatch.worker: batch.")
        return runtime.to_dict()

    def as_dict(self):
        return self.serialise_search(self._init_params)

    def _checkpoint(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "population.json"
        signature = payload_digest(json.loads(encode(self.as_dict())))
        if path.exists():
            state = json.loads(path.read_text())
            if state.get("version") != 2 or state["signature"] != signature:
                raise ValueError("SA configuration changed; use a new directory.")
        else:
            if (self.directory / "candidates.db").exists():
                raise ValueError("Missing population checkpoint; use a new directory.")
            state = dict(version=2, signature=signature, completed=-1, plans={},
                         random_states=self.random_streams.snapshot())
            write_json(path, state)
        self.random_streams.restore(state["random_states"])
        return state

    def _save(self, state):
        state["random_states"] = self.random_streams.snapshot()
        write_json(self.directory / "population.json", state)

    def _cooling(self, generation, frames, seed):
        runtime = self.worker.as_dict()
        params = resolve_executor_parameters(runtime["executor"]["parameters"], "md")
        start, end = self.temperatures[generation - 1]
        params.update(temp=start, tend=end, velocity_seed=None,
                      ignore_atoms_velocities=True)
        runtime["executor"]["parameters"] = params
        worker = create_worker(runtime, directory=self.directory / f"gen{generation}" / "sim")
        worker.skip_failed_candidates = True
        return worker

    def _evaluate_stage(self, worker, frames, seeds, indices, state, plan, stage):
        if not frames:
            return {}
        worker.skip_failed_candidates = True
        # Explicit task directories keep snapshot identity independent of the
        # worker's single-calculation layout and its failure bookkeeping.
        worker.task_directories = [f"cand{i}" for i in range(len(frames))]
        worker.run(frames, rng_states=seeds)
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs():
            return None
        results = worker.retrieve(include_retrieved=True, use_archive=self.use_archive)
        failed = worker.get_failed_candidates()
        expected = {f"cand{i}" for i in range(len(frames))}
        by_name = {}
        for trajectory in results:
            if not trajectory:
                raise RuntimeError(f"Empty SA {stage} result.")
            atoms = trajectory[-1]
            name = atoms.info["wdir"]
            if name not in expected or name in by_name or name in failed:
                raise RuntimeError(f"Unexpected or duplicate SA {stage} result: {name}.")
            by_name[name] = trajectory
        if set(by_name) | set(failed) != expected:
            raise RuntimeError(f"Missing or duplicate SA {stage} results.")
        # A partially propagated MD trajectory is not a completed cooling attempt.
        for name, trajectory in list(by_name.items()):
            atoms = trajectory[-1]
            if atoms.info.get("earlystop"):
                failed[name] = {"message": str(atoms.info["earlystop"])}
                del by_name[name]
        plan.setdefault("failures", {})[stage] = {
            str(indices[int(name[4:])]): failure for name, failure in failed.items()
        }
        self._save(state)
        for name, failure in failed.items():
            self._print(f"SA {stage}: skipping candidate {indices[int(name[4:])]}: "
                        f"{failure.get('message', 'calculation failed')}")
        return {indices[i]: by_name[f"cand{i}"] for i in range(len(frames))
                if f"cand{i}" in by_name}

    def _select_snapshots(self, generation, trajectories, plan):
        """Persist selector output before publishing the selection manifest."""
        directory = self.directory / f"gen{generation}" / "sel"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "selection.json"
        selected_path = directory / "selected.xyz"
        if manifest.exists():
            records = json.loads(manifest.read_text())["snapshots"]
            if not selected_path.exists():
                raise RuntimeError("SA selection.json requires the saved selected.xyz.")
            frames = read(selected_path, ":") if records else []
            if len(frames) != len(records):
                raise RuntimeError("SA selected structures do not match selection.json.")
            return frames, records
        frames, records = [], []
        for replica, trajectory in sorted(trajectories.items()):
            selector = create_selector(copy.deepcopy(self.strategy_config["minimisation"]["selector"]))
            selector.directory = directory / f"replica{replica}"
            # Composed selectors own child RNGs; seed those as well so a
            # selection interrupted before its cache is written is reproducible.
            pending = [selector]
            seed_rng = np.random.default_rng(np.random.SeedSequence([plan["seed"], replica]))
            while pending:
                component = pending.pop(0)
                component.set_rng(seed=int(seed_rng.integers(0, 2**31 - 1)))
                pending.extend(getattr(component, "selectors", []))
            data = AtomsNDArray(trajectory)
            selector.select(data)
            for (frame_index,) in np.argwhere(data.markers):
                frame_index = int(frame_index)
                source = trajectory[frame_index]
                record = dict(replica=replica, frame_index=frame_index,
                              parent=plan["parents"][replica],
                              md_step=int(source.info.get("step", -1)))
                atoms = source.copy()
                atoms.info = {}
                atoms.calc = None
                if "momenta" in atoms.arrays:
                    del atoms.arrays["momenta"]
                frames.append(atoms)
                records.append(record)
        temporary = directory / "selected.tmp.xyz"
        if frames:
            write(temporary, frames, format="extxyz")
        else:
            temporary.write_text("")
        os.replace(temporary, selected_path)
        write_json(manifest, {"snapshots": records})
        return frames, records

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
            seeds = np.random.default_rng(plan["seed"]).integers(0, 2**31 - 1, len(frames)).tolist()
            indices = list(range(len(frames)))
            records = [dict(replica=i) for i in indices]
            if generation:
                manifest = root / "sel" / "selection.json"
                if manifest.exists():
                    frames, records = self._select_snapshots(generation, {}, plan)
                else:
                    dynamics = self._cooling(generation, frames, plan["seed"])
                    cooled = self._evaluate_stage(
                        dynamics, frames, seeds, indices, state, plan, "dynamics")
                    if cooled is None:
                        return
                    frames, records = self._select_snapshots(generation, cooled, plan)
                indices = list(range(len(records)))
                seeds = [int(np.random.SeedSequence(
                    [plan["seed"], record["replica"], record["frame_index"]]
                ).generate_state(1)[0]) for record in records]
            worker = create_worker(self._minimum_runtime(), directory=root / "min")
            minima = self._evaluate_stage(
                worker, frames, seeds, indices, state, plan, "minimisation")
            if minima is None:
                return
            for index, trajectory in minima.items():
                atoms = trajectory[-1]
                record = records[index]
                replica = record["replica"]
                canonical_candidates_from_worker_results(
                    [atoms], gen_num=generation, objective=self.objective,
                    extinct_callbacks=self.population_config.extinct_callbacks)
                atoms.info["data"].update(parents=[plan["parents"][replica]] if generation else [],
                                          cycle=generation, replica=replica)
                evaluation_key = f"sa:{generation}:{replica}"
                if generation:
                    atoms.info["data"].update(frame_index=record["frame_index"], md_step=record["md_step"])
                    evaluation_key += f":{record['frame_index']}"
                database.add_evaluated_candidate(atoms, evaluation_key)
            self.population.refresh(database)
            if not self.population.candidates:
                raise RuntimeError("SA has no surviving candidates to continue exploration.")
            plan["retained"] = [a.info["confid"] for a in self.population.candidates]
            state["completed"] = generation
            self._save(state)
            self._print(f"SA generation {generation}: evaluated {len(minima)}, retained {len(plan['retained'])}")
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
                worker = create_worker(self._minimum_runtime(), directory=root / "min")
                worker.skip_failed_candidates = True
                workers.append(worker)
        return workers
