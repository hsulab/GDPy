#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools

from ase import Atoms
from ase.io import write

from gdpx.execution.factory import create_worker
from gdpx.execution.workers.drive import DriverBasedWorker
from gdpx.execution.workers.single import SingleWorker

from .monte_carlo import MCStepState, MonteCarlo
from ..move_step import _store_pending, read_pending, run_worker_move
from ..checkpoint import read_snapshot
from ..sampling import parse_operators

MC_EARLYSTOP_FNAME = "MC_EARLY_STOPPED"


class HybridMonteCarlo(MonteCarlo):
    def __init__(self, procedure, num_mcmoves: int, extra_workers={}, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.procedure = procedure

        self.num_mcmoves = num_mcmoves

        self.extra_workers = extra_workers

        return

    def _parse_procedure(self):
        """Parse the procedure for workers and steps."""
        prototype_workers, procedure_steps = [], []
        for subprocedure in self.procedure:
            if isinstance(subprocedure, list):
                assert len(subprocedure) == 2 and subprocedure[0] == "monte_carlo", ""
                worker_name = subprocedure[1].split("_")[1]
                runtime_config = self.extra_workers.get(worker_name, None)
                if runtime_config is not None:
                    subworker = create_worker(runtime_config)
                    if isinstance(subworker, DriverBasedWorker):
                        self._print("Convert a DriverBasedWorker to a SingleWorker.")
                        subworker = SingleWorker.from_a_worker(subworker)
                    assert isinstance(subworker, SingleWorker), (
                        f"{self.__class__.__name__} only supports SingleWorker (set use_single=True) but {subprocedure} is not."
                    )
                    subworker.directory = self.directory / "mc"
                    subproc_func = functools.partial(self._irun_metropolis, worker=subworker)
                    procedure_steps.append(("mc", subproc_func))
                    prototype_workers.append(subworker)
                else:
                    raise RuntimeError(f"Unknown subprocedure with worker {subprocedure}.")
            elif subprocedure.startswith("worker"):
                worker_name = subprocedure.split("_")[1]
                runtime_config = self.extra_workers.get(worker_name, None)
                if runtime_config is not None:
                    subworker = create_worker(runtime_config)
                    assert subworker is not None, f"Unknown worker {worker_name} in extra_workers."
                    # if isinstance(subworker, DriverBasedWorker):
                    #     self._print("Convert a DriverBasedWorker to a SingleWorker.")
                    #     subworker = SingleWorker.from_a_worker(subworker)
                    # assert isinstance(
                    #     subworker, SingleWorker
                    # ), f"{self.__class__.__name__} only supports SingleWorker (set use_single=True) but {subprocedure} is not."
                    subworker.directory = self.directory / worker_name
                    subproc_func = functools.partial(self._irun_dynamics, worker=subworker)
                    procedure_steps.append((worker_name, subproc_func))
                    prototype_workers.append(subworker)
                else:
                    raise RuntimeError(f"Unknown subprocedure with worker {subprocedure}.")
            else:
                raise RuntimeError(f"Unknown subprocedure {subprocedure}.")

        return procedure_steps, prototype_workers

    def _run(self):
        """"""
        # set init worker
        self.worker.directory = self.directory / "calculations" / "step.0000"

        # Format indent
        for op in self.operators:
            op.indent = "  "

        # check if subprocedures in the procedure are all valid
        procedure_steps, self._protype_workers = self._parse_procedure()

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

    def _irun_metropolis(self, step: int, name: str, worker: SingleWorker) -> MCStepState:
        """Run a sequence of proposals, resuming a pending attempt without redrawing."""
        directory = self._procedure_directory(step)
        context = getattr(self, "_resume_context", None)
        start = context["index"] if context else 0
        for i in range(start, self.num_mcmoves):
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
            self._save_step_info(result.operator, (step - 1) * self.num_mcmoves + i,
                                 result.accepted, self.energy_stored, result.energy, result.diagnostic)
            if result.accepted:
                self.energy_stored = result.energy
            state = self._check_earlystop(self.atoms)
            if state == MCStepState.EARLYSTOPPED:
                return state
        return MCStepState.FINISHED

    def as_dict(self) -> dict:
        """Return a dictionary representation of the object."""
        common = super().as_dict()
        recipe = common["recipe"]
        # Hybrid Monte Carlo has not migrated to the recipe input yet. Keep its
        # existing flat serialization until that method adopts the shared schema.
        d = {
            "method": "hybrid_monte_carlo",
            **recipe,
            "runtime": common["runtime"],
        }
        d.update(
            {
                "procedure": self.procedure,
                "num_mcmoves": self.num_mcmoves,
                "extra_workers": self.extra_workers,
            }
        )
        return d


if __name__ == "__main__":
    ...
