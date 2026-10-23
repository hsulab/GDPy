"""Restartable ordered execution on one fixed potential energy surface."""
import copy
import shutil
from io import StringIO

import numpy as np
from ase import Atoms
from ase.io import read, write
from gdpx.core.output import candidate_logging, quiet_logging

from gdpx.execution.accepted_state import load_structure, save_structure
from gdpx.execution.checkpoint import load_data, publish_snapshot, read_snapshot, save_data
from gdpx.execution.executor import ExecutionSettings, RunState, SingleStructureExecutor
from gdpx.execution.factory import create_worker
from gdpx.execution.fingerprint import payload_digest
from gdpx.providers.specs import thaw


class CompositeExecutor(SingleStructureExecutor):
    """Commit child boundaries independently of the public trajectory cadence."""

    def __init__(self, config, children, initializer, convergence, dump_period=1,
                 ckpt_period=100, ckpt_keep=2):
        parameters = thaw(config.executor.parameters)
        super().__init__(random_seed=parameters.get("random_seed"))
        self.runtime_config = config.to_dict()
        self.name = config.executor.method
        self.setting = ExecutionSettings(task=self.name)
        self.children = children
        self.initializer = initializer
        self.convergence = convergence
        self.dump_period, self.ckpt_period, self.ckpt_keep = dump_period, ckpt_period, ckpt_keep
        self.TRAJ_NAME = "mc.xyz"
        self.signature = payload_digest(self.runtime_config)

    def reset(self):
        pass  # All invocation state is persisted in this candidate's directory.

    def as_dict(self):
        return copy.deepcopy(self.runtime_config)

    def _load(self, path):
        state = load_data(path / "state.json")
        if state.get("version") != 1 or state.get("signature") != self.signature:
            raise ValueError("Composite input/checkpoint changed; start a new run.")
        return state, load_structure(path / "structure.json")

    def _commit(self, state, atoms, suffix=""):
        state["rng"] = self.rng.bit_generator.state
        path = self.directory / self.TRAJ_NAME
        state["trajectory_size"] = path.stat().st_size if path.exists() else 0
        staging = self.directory / "staging-composite"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        save_structure(staging / "structure.json", atoms,
                       copy.deepcopy(atoms.calc.results) if atoms.calc is not None else None)
        save_data(staging / "state.json", state)
        destination = self.directory / f"checkpoint.{state['step']}{suffix}"
        publish_snapshot(self.directory, staging, destination, "checkpoint.", keep=self.ckpt_keep)

    def _run_child(self, runtime, atoms, seed, directory):
        worker = create_worker(runtime.config, directory=directory)
        worker.run([atoms], rng_states=[seed])
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs():
            return None
        trajectories = worker.retrieve(include_retrieved=True)
        if len(trajectories) != 1 or not trajectories[0]:
            raise RuntimeError("Composite child requires exactly one completed trajectory.")
        result = trajectories[0][-1]
        # Invocation metadata must not become the next child's structural identity.
        result.info = {key: value for key, value in atoms.info.items()
                       if key not in {"step", "wdir", "confid", "group", "earlystop"}}
        if len(result) == len(atoms) and not result.constraints:
            result.set_constraint(copy.deepcopy(atoms.constraints))
        return result

    def _seed(self, state, index):
        if state["seed"] is None:
            runtime = self.initializer if index == -1 else self.children[index]
            child_seed = runtime.config.executor.parameters.get("random_seed")
            entropy = [int(self.rng.integers(0, 2**31)), state["step"], index + 1]
            if isinstance(child_seed, int):
                entropy.append(child_seed)
            state["seed"] = int(np.random.SeedSequence(entropy).generate_state(1)[0])
        return state["seed"]

    def _earlystop(self, atoms):
        stop = self.convergence.get("earlystop")
        if stop:
            if stop.get("property") != "energy_per_atom":
                raise ValueError("Composite earlystop supports energy_per_atom.")
            lower, upper = stop["range"]
            return not lower <= atoms.get_potential_energy() / len(atoms) < upper
        return False

    def run(self, atoms, read_ckpt=True, **kwargs):
        with candidate_logging(self.directory), quiet_logging():
            return self._run(atoms)

    def _run(self, atoms):
        if not isinstance(atoms, Atoms):
            raise TypeError("Composite executors require one Atoms input per chain.")
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.name == "hybrid_monte_carlo":
            from gdpx.execution.monte_carlo.output import report_hybrid_intro
            report_hybrid_intro([child.config.executor.to_dict() for child in self.children],
                                self.convergence["steps"], self.random_seed, self.TRAJ_NAME)
        path = self.directory / self.TRAJ_NAME
        if (self.directory / "current.json").exists():
            _, (state, atoms) = read_snapshot(self.directory, self._load)
            self.rng.bit_generator.state = state["rng"]
            if path.exists():
                with path.open("r+b") as stream:
                    stream.truncate(state["trajectory_size"])
        else:
            if path.exists() or any(self.directory.glob("checkpoint.*")):
                raise ValueError("Missing composite checkpoint; start a new run.")
            atoms = atoms.copy()
            state = dict(version=1, signature=self.signature, step=0, child=-1,
                         seed=None, initialized=False, early_stopped=False)
        if not state["initialized"]:
            seed = self._seed(state, -1)
            self._commit(state, atoms, "-initial")
            result = self._run_child(self.initializer, atoms, seed,
                                     self.directory / "steps" / "step.0000")
            if result is None:
                return RunState.PENDING
            atoms = result
            atoms.info["step"] = 0
            write(path, atoms)
            state.update(initialized=True, child=0, seed=None)
            self._commit(state, atoms)
        while state["step"] < self.convergence["steps"] and not state["early_stopped"]:
            while state["child"] < len(self.children):
                index = state["child"]
                seed = self._seed(state, index)
                self._commit(state, atoms, f"-child.{index}")
                directory = (self.directory / "steps" / f"step.{state['step'] + 1:04d}"
                             / f"procedure.{index:04d}")
                result = self._run_child(self.children[index], atoms, seed, directory)
                if result is None:
                    return RunState.PENDING
                atoms = result
                state.update(child=index + 1, seed=None)
                self._commit(state, atoms, f"-child.{index + 1}")
            state.update(step=state["step"] + 1, child=0)
            state["early_stopped"] = self._earlystop(atoms)
            if (state["step"] % self.dump_period == 0 or state["step"] == self.convergence["steps"]
                    or state["early_stopped"]):
                atoms.info["step"] = state["step"]
                write(path, atoms, append=True)
            self._commit(state, atoms)
        if state["early_stopped"]:
            (self.directory / "MC_EARLY_STOPPED").touch()
        return RunState.FINISHED

    def read_convergence(self):
        if not (self.directory / "current.json").exists():
            return False
        _, (state, _) = read_snapshot(self.directory, self._load, repair=False)
        return state["initialized"] and (state["early_stopped"] or state["step"] >= self.convergence["steps"])

    def read_trajectory(self, add_step_info=False, archive_path=None, **kwargs):
        path = self.directory / self.TRAJ_NAME
        if archive_path is None:
            return read(path, ":") if path.exists() else []
        from gdpx.utils.archive import open_archive
        with open_archive(archive_path) as archive:
            for member in archive:
                if member.name == f"{self.directory.name}/{self.TRAJ_NAME}":
                    return read(StringIO(archive.extractfile(member).read().decode()), ":", format="extxyz")
        return []

    def get_workers(self):
        from gdpx.execution.workers.metadata import WorkerMetadata
        workers = []
        for path in sorted((self.directory / "steps").glob("**/_meta/inputs.json")):
            root = path.parent.parent
            metadata = WorkerMetadata(root)
            definition = metadata.calculation_set(metadata.inputs.read(), ".")
            workers.append(create_worker(definition["batches"][0]["runtime"], directory=root))
        return workers
