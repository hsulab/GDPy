#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import functools
from collections.abc import Mapping

from ase import Atoms
from ase.io import write

from gdpx.execution.factory import create_worker
from gdpx.execution.workers.drive import DriverBasedWorker
from gdpx.execution.workers.single import SingleWorker

from .monte_carlo import MCStepState, MonteCarlo, resolve_monte_carlo_system
from ..move_step import _store_pending, read_pending, run_worker_move
from ..checkpoint import read_snapshot
from ..sampling import parse_operators
from .output import report_hybrid_intro

MC_EARLYSTOP_FNAME = "MC_EARLY_STOPPED"


def _runtime_executor_method(runtime, path):
    if not isinstance(runtime, Mapping):
        raise TypeError(f"{path} must be a runtime mapping.")
    executor = runtime.get("executor")
    if not isinstance(executor, Mapping) or not isinstance(executor.get("method"), str):
        raise ValueError(f"{path} requires executor.method.")
    return executor["method"]


def create_hybrid_monte_carlo(system, strategy, random_seed=None, directory="./"):
    """Create hybrid MC from a structured system and explicit cycle stages."""
    if not isinstance(strategy, Mapping):
        raise TypeError("hybrid_monte_carlo strategy must be a mapping.")
    unknown = strategy.keys() - {"operators", "cycle", "steps", "earlystop", "ckpt_period"}
    if unknown:
        raise ValueError(
            f"Unsupported hybrid_monte_carlo strategy settings: {', '.join(sorted(unknown))}."
        )
    resolved_operators, ensemble = resolve_monte_carlo_system(
        system, strategy.get("operators"), "hybrid_monte_carlo"
    )

    cycles = strategy.get("steps", 1)
    ckpt_period = strategy.get("ckpt_period", 100)
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 0:
        raise ValueError("strategy.steps must be a non-negative integer.")
    if not isinstance(ckpt_period, int) or isinstance(ckpt_period, bool) or ckpt_period <= 0:
        raise ValueError("strategy.ckpt_period must be a positive integer.")

    cycle = strategy.get("cycle")
    if not isinstance(cycle, list) or not cycle:
        raise ValueError("hybrid_monte_carlo requires a nonempty strategy.cycle list.")
    resolved_cycle = copy.deepcopy(cycle)
    for index, stage in enumerate(resolved_cycle):
        path = f"strategy.cycle.{index}"
        if not isinstance(stage, Mapping):
            raise TypeError(f"{path} must be a mapping.")
        method = stage.get("method")
        if method == "molecular_dynamics":
            unknown = stage.keys() - {"method", "runtime"}
            expected_executor = "md"
        elif method == "monte_carlo":
            unknown = stage.keys() - {"method", "runtime", "steps"}
            expected_executor = "spc"
            moves = stage.get("steps")
            if not isinstance(moves, int) or isinstance(moves, bool) or moves < 0:
                raise ValueError(f"{path}.steps must be a non-negative integer.")
        else:
            raise ValueError(
                f"{path}.method must be molecular_dynamics or monte_carlo."
            )
        if unknown:
            raise ValueError(f"Unsupported {path} settings: {', '.join(sorted(unknown))}.")
        actual_executor = _runtime_executor_method(stage.get("runtime"), f"{path}.runtime")
        if actual_executor != expected_executor:
            raise ValueError(
                f"{path} with method {method} requires runtime.executor.method: "
                f"{expected_executor}, got {actual_executor!r}."
            )

    convergence = {"steps": cycles}
    if "earlystop" in strategy:
        convergence["earlystop"] = copy.deepcopy(strategy["earlystop"])
    engine = HybridMonteCarlo(
        builder=system["builder"], operators=resolved_operators,
        convergence=convergence, cycle=resolved_cycle,
        random_seed=random_seed, dump_period=1, ckpt_period=ckpt_period,
        ignore_atoms_tags=system.get("ignore_atoms_tags", True),
        should_retry=False, restart=False, directory=directory,
    )
    engine.system_config = {
        "ensemble": ensemble,
        "ignore_atoms_tags": system.get("ignore_atoms_tags", True),
    }
    engine.strategy_config = copy.deepcopy(dict(strategy))
    return engine


class HybridMonteCarlo(MonteCarlo):
    runtime_method_name = "hybrid_monte_carlo"
    INFO_NAME = "mcmoves.log"

    def __init__(self, cycle, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self.cycle = copy.deepcopy(cycle)

    def _parse_cycle(self):
        """Parse one cycle into workers and executable stages."""
        prototype_workers, procedure_steps = [], []
        moves_per_cycle = sum(
            stage.get("steps", 0) for stage in self.cycle
            if stage["method"] == "monte_carlo"
        )
        move_offset = 0
        for stage in self.cycle:
            subworker = create_worker(stage["runtime"])
            method = stage["method"]
            if method == "molecular_dynamics":
                assert isinstance(subworker, DriverBasedWorker)
                subproc_func = functools.partial(self._irun_dynamics, worker=subworker)
                procedure_steps.append((method, subproc_func))
            else:
                assert isinstance(subworker, SingleWorker)
                subproc_func = functools.partial(
                    self._irun_metropolis, worker=subworker,
                    num_mcmoves=stage["steps"], move_offset=move_offset,
                    moves_per_cycle=moves_per_cycle,
                )
                procedure_steps.append((method, subproc_func))
                move_offset += stage["steps"]
            prototype_workers.append(subworker)

        return procedure_steps, prototype_workers

    def _run(self):
        """"""
        # set init worker
        self.worker.directory = self.directory / "calculations" / "step.0000"

        # Format indent
        for op in self.operators:
            op.indent = "  "

        # check if subprocedures in the procedure are all valid
        procedure_steps, self._protype_workers = self._parse_cycle()
        report_hybrid_intro(
            self.cycle, self.convergence["steps"], self.random_seed,
            self.TRAJ_NAME, self.INFO_NAME,
        )

        # enter the main loop
        converged = self.read_convergence()
        if not converged:
            # init structure
            step_converged = False
            self._resume_context = None
            self._cleanup_committed_pending()
            if (self.directory / "pending-hybrid").exists():
                self.atoms, pending_data = read_pending(self.directory / "pending-hybrid", self.rng)
                self.energy_stored = pending_data["energy"]
                self._resume_context = pending_data["context"]
                self.start_step = self._resume_context["step"] - 1
                step_converged = True
            elif not self._verify_checkpoint():
                step_converged = self._init_structure()
            else:
                step_converged = True
                self._load_checkpoint()

            if not step_converged:
                self._print("Wait structure to initialise.")
                return
            else:
                self.start_step += 1

            # run procedure
            curr_step = self.start_step
            while True:
                if curr_step > self.convergence["steps"]:
                    self._print("Monte Carlo reaches the maximum step.")
                    break
                if (self.directory / MC_EARLYSTOP_FNAME).exists():
                    self._print("Monte Carlo reaches the earlystopping.")
                    break

                step_state = MCStepState.UNFINISHED
                self._print(f"===== Hybrid MC Step {curr_step} =====")
                for procedure_index, (subproc_name, subproc_func) in enumerate(procedure_steps):
                    if self._resume_context and procedure_index < self._resume_context["procedure_index"]:
                        continue
                    self._procedure_index = procedure_index
                    step_state = subproc_func(name=subproc_name, step=curr_step)
                    if step_state == MCStepState.UNFINISHED:
                        self._print("Wait MC step to finish.")
                        break
                    elif step_state == MCStepState.FINISHED:
                        # post worker compute with MC
                        ...
                    elif step_state == MCStepState.FAILED:
                        # This should not happen as many mcmoves are done consecutively.
                        self._print(f"RETRY STEP {curr_step}.")
                        break
                    elif step_state == MCStepState.EARLYSTOPPED:
                        # We need a file flag to indicate the simutlation is finshed
                        # when read_convergence is called.
                        with open(self.directory / MC_EARLYSTOP_FNAME, "w") as fopen:
                            fopen.write(f"{step_state =}")
                        break
                    else:
                        ...
                else:
                    # One trajectory frame represents one complete procedure cycle.
                    write(self.directory / self.TRAJ_NAME, self.atoms, append=True)
                    self._save_checkpoint(curr_step)
                    curr_step += 1
                if step_state != MCStepState.FINISHED:
                    break
        else:
            self._print("Monte Carlo is converged.")

        return

    def _load_checkpoint(self):
        """Resume a completed hybrid procedure without rewinding substep workers."""
        self._validate_calculation_layout()
        if not (self.directory / "current.json").exists():
            raise ValueError("Legacy hybrid checkpoint is not supported; start a new run.")
        _, (state, self.atoms) = read_snapshot(self.directory, self._read_snapshot)
        self.operators, self.op_probs = parse_operators(state["operators"])
        self._attach_bond_length_minimum_list()
        self.rng.bit_generator.state = state["rng"]
        self.start_step = state["step"]
        self.energy_stored = self.atoms.get_potential_energy()
        for name, size in state["output_sizes"].items():
            with (self.directory / name).open("r+b") as stream:
                stream.truncate(size)
        self._prune_calculations(self.start_step)

    def _procedure_directory(self, step):
        return (self.directory / "calculations" / f"step.{step:04d}"
                / f"procedure.{getattr(self, '_procedure_index', 0):04d}")

    def _irun_dynamics(self, step: int, name: str, worker: DriverBasedWorker) -> MCStepState:
        """"""
        self._print(f">>>>> {name.upper()} ")
        worker.directory = self._procedure_directory(step) / "excurs"
        context = dict(step=step, procedure_index=getattr(self, "_procedure_index", 0), kind="dynamics")
        pending_path = self.directory / "pending-hybrid"
        if getattr(self, "_resume_context", None) == context:
            self.atoms, saved = read_pending(pending_path, self.rng)
            self.energy_stored = saved["energy"]
            if saved.get("resolved"):
                self._resume_context = None
                return self._check_earlystop(self.atoms)
        data = dict(version=2, context=context, energy=self.energy_stored,
                    rng=self.rng.bit_generator.state, resolved=False)
        _store_pending(pending_path, self.atoms, data)

        # Get tags as it is not stored by the worker.
        curr_atoms = self.atoms
        curr_tags = curr_atoms.get_tags()

        _ = worker.run([curr_atoms])
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            curr_atoms: Atoms = worker.retrieve(include_retrieved=True)[0][-1]
            curr_atoms.set_tags(curr_tags)

            self.energy_operated = curr_atoms.get_potential_energy()
            self._print(f"  ene {self.energy_stored:>18.4f} -> {self.energy_operated:>18.4f}")

            self.energy_stored = self.energy_operated
            self.atoms = curr_atoms
            data.update(energy=self.energy_stored, resolved=True)
            _store_pending(pending_path, self.atoms, data)
            self._resume_context = None

            step_state = self._check_earlystop(self.atoms)
        else:
            step_state = MCStepState.UNFINISHED

        return step_state

    def _irun_metropolis(
        self, step: int, name: str, worker: SingleWorker,
        num_mcmoves: int, move_offset: int = 0, moves_per_cycle: int = 0,
    ) -> MCStepState:
        """Run a sequence of proposals, resuming a pending attempt without redrawing."""
        directory = self._procedure_directory(step)
        context = getattr(self, "_resume_context", None)
        start = context["index"] if context else 0
        for i in range(start, num_mcmoves):
            worker.directory = directory / f"proposal.{i:04d}"
            worker.wdir_name = "cand0"
            result = run_worker_move(
                self.atoms, self.energy_stored, self.operators, self.op_probs, self.rng,
                worker, self.directory / "pending-hybrid",
                info={"confid": step, "step": -1},
                resume_context={"step": step, "index": i,
                                "procedure_index": getattr(self, "_procedure_index", 0)},
            )
            self.atoms = result.atoms
            if result.accepted is None:
                self.energy_stored = result.energy
                return MCStepState.UNFINISHED
            self._resume_context = None
            attempt = (step - 1) * moves_per_cycle + move_offset + i
            self._save_step_info(result.operator, attempt,
                                 result.accepted, self.energy_stored, result.energy, result.diagnostic)
            if result.accepted:
                self.energy_stored = result.energy
            state = self._check_earlystop(self.atoms)
            if state == MCStepState.EARLYSTOPPED:
                return state
        return MCStepState.FINISHED

    def as_dict(self) -> dict:
        """Return the public structured hybrid MC configuration."""
        potential = self.worker.runtime.provider_potential
        if hasattr(potential, "remove_loaded_models"):
            potential.remove_loaded_models()
        system = copy.deepcopy(self.system_config)
        system["builder"] = self.builder.as_dict()
        strategy = copy.deepcopy(self.strategy_config)
        strategy["steps"] = self.convergence["steps"]
        if "earlystop" in self.convergence:
            strategy["earlystop"] = copy.deepcopy(self.convergence["earlystop"])
        strategy["ckpt_period"] = self.ckpt_period
        strategy["cycle"] = copy.deepcopy(self.cycle)
        return {
            "method": "hybrid_monte_carlo",
            "random_seed": self.random_seed,
            "system": system,
            "strategy": strategy,
            "runtime": self.worker.as_dict(),
        }


if __name__ == "__main__":
    ...
