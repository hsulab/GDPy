import copy
import enum
import shutil
from collections.abc import Mapping

import numpy as np
from ase import Atoms, data
from ase.io import read, write

from gdpx.core.output import candidate_logging, quiet_logging
from gdpx.providers.ensemble import resolve_ensemble
from gdpx.execution.workers.drive import DriverBasedWorker
from gdpx.execution.workers.single import SingleWorker

from gdpx.execution.executor import SingleStructureExecutor, ExecutionSettings, RunState
from gdpx.execution.move_step import read_pending, run_worker_move
from gdpx.execution.accepted_state import load_accepted_state, save_accepted_state
from gdpx.execution.checkpoint import load_data, save_data, publish_snapshot, read_snapshot, prune_snapshots
from gdpx.sampling.moves.operator import BaseMCOperator
from gdpx.sampling import parse_operators
from gdpx.sampling.geometry import infer_unique_atomic_numbers, prepare_operators
from gdpx.execution.monte_carlo.output import mc_box, report_outcome, report_setup, report_status

"""This module tries to offer a base class for all MonteCarlo-like methods.
"""

MC_EARLYSTOP_FNAME = "MC_EARLY_STOPPED"

MCStepState = enum.Enum("MCStepState", ("UNFINISHED", "FINISHED", "FAILED", "EARLYSTOPPED"))


_EXCHANGE_OPERATORS = {
    "exchange", "biased_volume_exchange", "cavity_exchange", "adsorbate_exchange",
}


def resolve_monte_carlo_system(setup, operators, method_name="monte_carlo"):
    """Validate a structured MC system and resolve ensemble operator settings."""
    if not isinstance(setup, Mapping):
        raise TypeError(f"{method_name} setup must be a mapping.")
    unknown = setup.keys() - {"ensemble", "ignore_atoms_tags"}
    if unknown:
        raise ValueError(f"Unsupported {method_name} setup settings: {', '.join(sorted(unknown))}.")
    if not isinstance(setup.get("ignore_atoms_tags", True), bool):
        raise TypeError("setup.ignore_atoms_tags must be a boolean.")
    if not isinstance(operators, list) or not operators:
        raise ValueError(f"{method_name} requires a nonempty setup.operators list.")

    ensemble = resolve_ensemble(setup.get("ensemble"), method_name)
    ensemble_method = ensemble["method"]
    resolved_operators = copy.deepcopy(operators)
    for operator in resolved_operators:
        if not isinstance(operator, Mapping):
            raise TypeError("Every setup operator must be a mapping.")

    temperature = ensemble["temperature"]
    chemical_potentials = ensemble.get("chemical_potentials", {})

    for operator in resolved_operators:
        if "temperature" in operator or "chempots" in operator:
            raise ValueError(
                "Move operator temperature and chempots to parameters.ensemble."
            )
        if method_name != "biased_monte_carlo" and operator.get("skip_distance_check", True) is not True:
            raise ValueError(
                "MC ensembles require skip_distance_check: true because "
                "distance-filtered proposals break detailed balance; use biased_monte_carlo for filtered MC proposals."
            )
        if method_name != "biased_monte_carlo":
            operator["skip_distance_check"] = True
        elif not isinstance(operator.get("skip_distance_check", False), bool):
            raise ValueError("skip_distance_check must be a boolean.")
        operator["temperature"] = float(temperature)
        name = operator.get("method", "move")
        particles = operator.get("particles", [])
        if name == "swap_type":
            if ensemble_method not in {"suvt", "uvt"}:
                raise ValueError("swap_type requires a suvt or uvt ensemble.")
            missing = [particle for particle in particles if particle not in chemical_potentials]
            if missing:
                raise ValueError(f"Missing chemical potentials for: {', '.join(missing)}.")
            operator["chempots"] = [chemical_potentials[particle] for particle in particles]
        elif name in _EXCHANGE_OPERATORS:
            if ensemble_method != "uvt":
                raise ValueError(f"{name} requires a uvt ensemble.")
            missing = [particle for particle in particles if particle not in chemical_potentials]
            if missing:
                raise ValueError(f"Missing chemical potentials for: {', '.join(missing)}.")
            operator["chempots"] = [chemical_potentials[particle] for particle in particles]
        elif name == "react":
            if ensemble_method != "uvt":
                raise ValueError("react requires a uvt ensemble.")
            reaction = operator.get("reaction", {})
            reaction_particles = reaction.get("particles", [])
            missing = [particle for particle in reaction_particles if particle not in chemical_potentials]
            if missing:
                raise ValueError(f"Missing chemical potentials for: {', '.join(missing)}.")
            if "chempot_0" in reaction:
                raise ValueError("Move reaction.chempot_0 to parameters.ensemble.chemical_potentials.")
            reaction["chempot_0"] = [chemical_potentials[particle] for particle in reaction_particles]

    return resolved_operators, copy.deepcopy(dict(ensemble))


def resolve_monte_carlo_convergence(convergence, method_name="monte_carlo"):
    """Validate and copy the public convergence settings shared by MC methods."""
    if not isinstance(convergence, Mapping):
        raise TypeError(f"{method_name} stop must be a mapping.")
    unknown = convergence.keys() - {"steps", "earlystop"}
    if unknown:
        raise ValueError(
            f"Unsupported {method_name} stop settings: "
            f"{', '.join(sorted(unknown))}."
        )
    steps = convergence.get("steps", 1)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("stop.steps must be a non-negative integer.")
    resolved = {"steps": steps}
    if "earlystop" in convergence:
        resolved["earlystop"] = copy.deepcopy(convergence["earlystop"])
    return resolved


def convert_blmin_to_str(blmin: dict) -> str:
    """"""
    elements = []
    for k in blmin.keys():
        elements.extend(k)
    elements = set(elements)
    nelements = len(elements)

    index_map = {}
    for i, e in enumerate(elements):
        index_map[e] = i
    distance_map = np.zeros((nelements, nelements))
    for (i, j), dis in blmin.items():
        distance_map[index_map[i], index_map[j]] = dis

    symbols = [data.chemical_symbols[e] for e in elements]

    content = "Bond Distance Minimum\n"
    # content += "  covalent ratio: {}\n".format(covalent_min)
    content += "  " + " " * 4 + ("{:>6}  " * nelements).format(*symbols) + "\n"
    for i, s in enumerate(symbols):
        content += "  " + ("{:<4}" + "{:>8.4f}" * nelements + "\n").format(s, *list(distance_map[i]))

    return content


class MonteCarlo(SingleStructureExecutor):
    name = "monte_carlo"
    restart = False
    required_evaluation_method = "spc"
    runtime_method_name = "monte_carlo"

    #: Prefix of the working directory.
    WDIR_PREFIX: str = "cand"

    #: Name of the MC trajectory.
    TRAJ_NAME: str = "mc.xyz"

    #: Name of the file stores MC information (operations).
    INFO_NAME: str = "opstat.txt"

    def __init__(
        self,
        operators: list[dict],
        convergence: dict,
        random_seed=None,
        dump_period: int = 1,
        ckpt_period: int = 100,
        ignore_atoms_tags: bool = True,
        should_retry: bool = True,
        restart: bool = False,
        directory="./",
    ) -> None:
        """Parameters for Monte Carlo.

        Args:
            ignore_atoms_tags: Whether ignore tags in atoms and set them by chemical symbols.

        """
        super().__init__(
            directory=directory,
            random_seed=random_seed,
        )

        self.dump_period = dump_period
        self.ckpt_period = ckpt_period

        self.ignore_atoms_tags = ignore_atoms_tags

        self.should_retry = should_retry

        self.restart = restart

        self.setting = ExecutionSettings(task=self.runtime_method_name)
        self.worker = None
        self._early_stopped = False

        # Parse operators
        self.operators, self.op_probs = parse_operators(operators)

        # Parse convergence
        self.convergence = convergence
        if self.convergence.get("steps", None) is None:
            self.convergence["steps"] = 1

        return

    def register_worker(self, worker, *args, **kwargs) -> None:
        """Attach the evaluation method required by this MC variant."""
        candidate = worker[0] if isinstance(worker, list) and worker else worker
        method = getattr(getattr(getattr(candidate, "runtime", None), "config", None), "executor", None)
        method = getattr(method, "method", None)
        if method != self.required_evaluation_method:
            raise ValueError(
                f"{self.runtime_method_name} requires runtime.executor.method: {self.required_evaluation_method}."
            )
        self.worker = candidate

    def _init_structure(self):
        with mc_box("initialization") as box:
            ready = self._evaluate_initial_structure()
            if ready:
                box.line(f"initial energy [eV]: {self.energy_stored:.6f} | atoms: {len(self.atoms)}")
            else:
                box.line("status: waiting for initial evaluation")
            return ready

    def _evaluate_initial_structure(self):
        """Initialise the input structure.

        Set proper tags and evaluate the structure. Prepare `self.atoms`,
        `self.energy_stored`, and `self.start_step`.

        """
        # Prepare the initial structure
        self.worker.directory = self.directory / "steps" / "step.0000"
        self._print("===== MonteCarlo Structure =====")
        tags = self.atoms.arrays.get("tags", None)
        if self.ignore_atoms_tags or tags is None:
            # default is setting tags by elements
            symbols = self.atoms.get_chemical_symbols()
            type_list = sorted(list(set(symbols)))
            new_tags = [type_list.index(s) * 10000 + i for i, s in enumerate(symbols)]
            self.atoms.set_tags(new_tags)
            self._print("set default tags by chemical symbols...")
        else:
            self._print("set attached tags from the structure...")

        # Evaluate with the selected runtime before any MC steps.
        self._print("===== MonteCarlo Initial Evaluation =====")
        # TODO: atoms lost tags in optimisation, and may move this part to driver?
        curr_tags = self.atoms.get_tags()

        self.atoms.info["confid"] = 0
        self.atoms.info["step"] = -1  # remove step info

        write(self.directory / "mc_attempts.xyz", self.atoms)

        # TODO: whether init driver?
        self.worker.wdir_name = "."
        _ = self.worker.run([self.atoms])
        self.worker.inspect(resubmit=True)
        if self.worker.get_number_of_running_jobs() == 0:
            curr_frames = self.worker.retrieve()[0]
            # - update atoms
            curr_atoms = curr_frames[-1]
            self.energy_stored = curr_atoms.get_potential_energy()
            self._print(f"ene: {self.energy_stored}")
            self.atoms = curr_atoms
            self.atoms.set_tags(curr_tags)
            write(self.directory / self.TRAJ_NAME, self.atoms)

            # -
            self.start_step = 0
            self.atoms.info["mc_step"] = 0

            # - log operator status
            with open(self.directory / self.INFO_NAME, "w") as fopen:
                fopen.write(
                    "{:<8s}  {:<24s}  {:<24s}  {:<12s}  {:<12s}  {:<24s}  {:<24s}  \n".format(
                        "#Step",
                        "Operator",
                        "Info",
                        "natoms",
                        "Success",
                        "prev_ene",
                        "curr_ene",
                    )
                )
            self._save_checkpoint(0, force=True)
            step_converged = True
        else:
            step_converged = False

        return step_converged

    def _attach_bond_length_minimum_list(self):
        """Find possible elements in the simulation and build a bond-distance list."""
        numbers = infer_unique_atomic_numbers(self.operators, substrates=[self.atoms])
        prepare_operators(self.operators, numbers)

        return

    def run(self, atoms, read_ckpt=True, **kwargs):
        """Run MonteCarlo simulation."""
        with candidate_logging(self.directory), quiet_logging():
            if not isinstance(atoms, Atoms):
                raise TypeError("Sampling executors require one Atoms input per chain.")
            self._run_with_worker(atoms)
            return RunState.FINISHED if self.read_convergence() else RunState.PENDING

    def _run_with_worker(self, atoms):
        self._validate_calculation_layout()
        if self.worker is None:
            raise RuntimeError("Sampling requires an evaluation worker.")

        # Check if it has a valid worker
        if isinstance(self.worker, DriverBasedWorker) and not isinstance(self.worker, SingleWorker):
            self._print("Convert a DriverBasedWorker to a SingleWorker.")
            self.worker = SingleWorker.from_a_worker(self.worker)
        assert isinstance(self.worker, SingleWorker), (
            f"{self.__class__.__name__} only supports SingleWorker (set use_single=True)."
        )
        self.worker.directory = self.directory

        self.atoms = atoms.copy()
        self._early_stopped = False

        # Prepare logger and output some basic info...
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        # Show operator information
        self._print("===== MonteCarlo Operators (Modifiers) =====")

        # Register bond list
        self._attach_bond_length_minimum_list()

        for op in self.operators:
            op._print = self._print
            op._debug = self._debug
            op.indent = "  "  # indent before any print or string
            for l in str(op).split("\n"):
                self._print(l)
            for l in convert_blmin_to_str(op.blmin).split("\n"):
                self._print("  " + l)
        self._print(f"normalised probabilities {self.op_probs}")
        report_setup(self.operators, self.op_probs, self.convergence["steps"], self.random_seed,
                     method=self.runtime_method_name)

        # NOTE: Something about statmech
        # Check if operators' regions are consistent
        # Though it works, unexpected results may occur.
        # TODO: need rewrite eq function as compare array is difficult
        # noperators = len(self.operators)
        # for i in range(1,noperators):
        #    if self.operators[i].region != self.operators[i-1].region:
        #        raise RuntimeError(f"Inconsistent region found in op {i-1} and op {i}")

        # For other MC methods, it only needs rewrite `_run` method.
        self._run()

        return

    def _run(self):
        """"""
        converged = self.read_convergence()
        if not converged:
            # Check if we start from scratch or restart from a checkpoint
            step_converged = False
            self._cleanup_committed_pending()
            pending = sorted(self.directory.glob("pending-move-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
            if pending:
                self.atoms, pending_data = read_pending(pending[-1], self.rng)
                self.energy_stored = pending_data["energy"]
                self.start_step = pending_data["context"]["step"] - 1
                step_converged = True
                report_status("resume", f"pending evaluation at step {self.start_step + 1}")
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

            # Run mc steps
            curr_step = self.start_step  # start_step
            while True:
                # -- check exit-loop conditions
                if curr_step > self.convergence["steps"]:
                    report_status("complete", f"step budget reached: {self.convergence['steps']} | "
                                  f"trajectory: {self.directory / self.TRAJ_NAME}")
                    break
                if (self.directory / MC_EARLYSTOP_FNAME).exists():
                    report_status("early stopped", f"stopped before step {curr_step} | "
                                  f"trajectory: {self.directory / self.TRAJ_NAME}")
                    break
                # -- run step
                step_state = self._irun(curr_step)
                # -- state-specific
                if step_state == MCStepState.UNFINISHED:
                    self._print("Wait MC step to finish.")
                    break
                elif step_state == MCStepState.FINISHED:
                    # -- save checkpoint
                    self._save_checkpoint(step=curr_step, force=curr_step == self.convergence["steps"])
                    # -- clean up
                    if ((self.directory / "steps" / f"step.{curr_step:04d}").exists()) and (
                        curr_step % self.dump_period != 0
                    ):
                        shutil.rmtree(self.directory / "steps" / f"step.{curr_step:04d}")
                    curr_step += 1
                elif step_state == MCStepState.FAILED:
                    self._print(f"RETRY STEP {curr_step}.")
                elif step_state == MCStepState.EARLYSTOPPED:
                    # We need a file flag to indicate the simutlation is finshed
                    # when read_convergence is called.
                    with open(self.directory / MC_EARLYSTOP_FNAME, "w") as fopen:
                        fopen.write(f"{step_state =}")
                else:
                    raise Exception(f"{step_state} should not happen.")
        else:
            status = "early stopped" if (self.directory / MC_EARLYSTOP_FNAME).exists() else "complete"
            report_status(status, f"existing run | trajectory: {self.directory / self.TRAJ_NAME}")

        return

    def _irun(self, istep: int) -> MCStepState:
        with mc_box(f"step {istep}/{self.convergence['steps']}") as box:
            return self._run_trial(istep, box)

    def _run_trial(self, istep, box) -> MCStepState:
        """Run a move without retaining a mutated accepted structure while waiting."""
        self._print(f"===== MC Step {istep} =====")
        self.worker.directory = self.directory / "steps" / f"step.{istep:04d}"
        self.worker.wdir_name = "."
        result = run_worker_move(
            self.atoms, self.energy_stored, self.operators, self.op_probs, self.rng,
            self.worker, self.directory / f"pending-move-{istep}",
            info={"confid": istep, "step": -1},
            attempts_path=self.directory / "mc_attempts.xyz",
            resume_context={"step": istep},
        )
        report_outcome(box, result, self.energy_stored)
        self.atoms = result.atoms
        if result.accepted is None:
            self.energy_stored = result.energy
            return MCStepState.UNFINISHED
        self.energy_operated = result.energy if result.valid else np.inf
        self._save_step_info(result.operator, istep, result.accepted,
                             self.energy_stored, self.energy_operated, result.diagnostic)
        if result.accepted:
            self.energy_stored = result.energy
        if not result.valid and self.should_retry:
            box.line("next action: retry this step")
            return MCStepState.FAILED
        if not result.valid:
            box.line("next action: retain current state and advance")
        self.atoms.info["mc_step"] = istep
        state = self._check_earlystop(self.atoms)
        if istep % self.dump_period == 0 or istep == self.convergence["steps"] or state == MCStepState.EARLYSTOPPED:
            write(self.directory / self.TRAJ_NAME, self.atoms, append=True)
        self._early_stopped = state == MCStepState.EARLYSTOPPED
        if (self.directory / f"pending-move-{istep}").exists() or self._early_stopped:
            self._save_checkpoint(istep, force=True)
        return state

    def _check_earlystop(self, atoms: Atoms) -> MCStepState:
        """Check whether earlystopping should be done to avoid unphysical structures.

        Returns:
            A state string `FINISHED` or `EARLYSTOPPED`.

        """
        # check MC earlystop convergence
        es_dict = self.convergence.get("earlystop", None)
        if es_dict is not None:
            prop = es_dict.get("property", None)
            if prop == "energy_per_atom":
                ae_min, ae_max = es_dict["range"]
                energy_per_atom = atoms.get_potential_energy() / len(atoms)
                if ae_min <= energy_per_atom < ae_max:
                    es_state = MCStepState.FINISHED
                else:
                    es_state = MCStepState.EARLYSTOPPED
                    self._print("MC earlystops by `energy_per_atom`.")
            else:
                raise NotImplementedError(f"Unknown earlystop: {es_dict =}.")
        else:
            es_state = MCStepState.FINISHED

        # check atoms earlystop convergence only works with ase driver
        earlystop = atoms.info.get("earlystop", False)
        if earlystop:
            es_state = MCStepState.EARLYSTOPPED
            self._print("MC earlystops by `driver observer`.")

        return es_state

    def _verify_checkpoint(self) -> bool:
        return (self.directory / "current.json").exists() or any(self.directory.glob("checkpoint.*"))

    def _cleanup_committed_pending(self):
        """Finish cleanup if publication succeeded before an interruption."""
        pending = list(self.directory.glob("pending-move-*"))
        if (self.directory / "pending-hybrid").exists():
            pending.append(self.directory / "pending-hybrid")
        if not pending or not (self.directory / "current.json").exists():
            return
        _, (state, _) = read_snapshot(self.directory, self._read_snapshot)
        for path in pending:
            data = load_data(path / "proposal.json")
            if data["context"]["step"] <= state["step"]:
                shutil.rmtree(path)

    def _save_checkpoint(self, step, force=False):
        pending_paths = list(self.directory.glob("pending-move-*"))
        if (self.directory / "pending-hybrid").exists():
            pending_paths.append(self.directory / "pending-hybrid")
        force = force or bool(pending_paths)
        if not force and not (self.ckpt_period > 0 and step % self.ckpt_period == 0):
            return
        destination = self.directory / f"checkpoint.{step}"
        if destination.exists() and not pending_paths:
            self._read_snapshot(destination)
            return
        staging = self.directory / f"staging-checkpoint-{step}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        save_accepted_state(staging / "structure.json", self.atoms, self.energy_stored)
        save_data(staging / "state.json", dict(version=2, step=step,
                  early_stopped=getattr(self, "_early_stopped", False),
                  operators=[op.as_dict() for op in self.operators], rng=self.rng.bit_generator.state,
                  output_sizes={name: (self.directory / name).stat().st_size
                                for name in (self.TRAJ_NAME, "mc_attempts.xyz", self.INFO_NAME)
                                if (self.directory / name).exists()}))
        publish_snapshot(self.directory, staging, destination, "checkpoint.",
                         keep=getattr(self, "ckpt_keep", 2))
        for pending in pending_paths:
            shutil.rmtree(pending)

    @staticmethod
    def _read_snapshot(path):
        state = load_data(path / "state.json")
        if state.get("version") != 2:
            raise ValueError("Unsupported MC checkpoint version; start a new run.")
        return state, load_accepted_state(path / "structure.json")

    def _load_checkpoint(self):
        """Load the current Monet Carlo checkpoint.

        We first infer the starting step from the checkpoint, then load saved
        operators, random state, and the structure, and finally reset the output
        files (mc.xyz, mc_attempts.xyz, and opstat.txt) to the starting step.
        Also, the computation folders beyond the checkpoint step will be removed.

        """
        self._validate_calculation_layout()
        if not (self.directory / "current.json").exists():
            raise ValueError("Legacy MC operator checkpoint is not supported; start a new run.")
        ckpt_wdir, (state, self.atoms) = read_snapshot(self.directory, self._read_snapshot)
        prune_snapshots(self.directory, "checkpoint.")
        step = state["step"]
        self.operators, self.op_probs = parse_operators(state["operators"])
        self._attach_bond_length_minimum_list()

        # Add print functions to operators
        for op in self.operators:
            op._print = self._print
            op._debug = self._debug

        self._print("<<<<< Saved MonteCarlo Operators (Modifiers) >>>>>")
        for op in self.operators:
            for x in str(op).split("\n"):
                self._print(x)
        self._print(f"normalised probabilities {self.op_probs}")

        # Load random state
        self._print("Load random state.")
        self.rng.bit_generator.state = state["rng"]

        # Load structure
        self._print("Load structure.")
        self.start_step = step
        self.energy_stored = self.atoms.get_potential_energy()
        report_status("resume", f"checkpoint step: {self.start_step} | "
                      f"current energy [eV]: {self.energy_stored:.6f} | atoms: {len(self.atoms)}")

        for name, size in state["output_sizes"].items():
            with (self.directory / name).open("r+b") as stream:
                stream.truncate(size)

        # Discard complete calculation folders beyond the committed checkpoint.
        self._prune_calculations(step)

        return

    def _save_step_info(self, curr_op: BaseMCOperator, istep: int, success: bool, prev_ene: float, curr_ene: float, extra_info: str = "-"):
        """Record an attempt without storing transient state on its operator."""

        num_atoms = len(self.atoms)
        with open(self.directory / self.INFO_NAME, "a") as fopen:
            fopen.write(
                f"{istep:<8d}  {curr_op.name:<24s}  {extra_info:<24s}  {num_atoms:<12d}  {str(success):<12s}  {prev_ene:<24.4f}  {curr_ene:<24.4f}  \n"
            )

        return

    def read_convergence(self):
        """Completion follows durable progress, independently of output cadence."""
        if (self.directory / MC_EARLYSTOP_FNAME).exists():
            return True
        if not (self.directory / "current.json").exists():
            return False
        _, (state, _) = read_snapshot(self.directory, self._read_snapshot, repair=False)
        return state.get("early_stopped", False) or state["step"] >= self.convergence["steps"]

    def _validate_calculation_layout(self):
        """Reject append-era runs before touching their checkpoints or outputs."""
        from gdpx.execution.workers.metadata import WorkerMetadata

        # The exploration CLI stores scheduler metadata at the run root. It
        # must not be confused with the old root-level calculation metadata.
        metadata = self.directory / "_meta"
        scheduler_metadata = (metadata / "_scheduler.json").is_file() and all(
            path.name == "_scheduler.json"
            or path.match("exp-*.json")
            or path.match("run.script-*")
            for path in metadata.iterdir()
        )
        compact_metadata = (metadata / "inputs.json").is_file()
        if compact_metadata:
            WorkerMetadata(self.directory).compact
        if ((metadata.exists() and not scheduler_metadata and not compact_metadata) or (self.directory / "_data").exists()
                or any(self.directory.glob("_*_jobs.json")) or any(self.directory.glob("cand*"))
                or any(self.directory.glob("step.*")) or (self.directory / "init" / "_meta").exists()):
            raise ValueError("Legacy append-based exploration layout; use a new working directory.")
        for path in (self.directory / "steps").glob("**/_meta"):
            WorkerMetadata(path.parent).compact

    def _prune_calculations(self, step):
        for path in (self.directory / "steps").glob("step.*"):
            if int(path.name.split(".")[1]) > step:
                shutil.rmtree(path)

    def get_workers(self):
        """Reconstruct workers from each retained calculation's saved runtime."""
        from gdpx.execution.factory import create_worker
        from gdpx.execution.workers.metadata import WorkerMetadata

        self._validate_calculation_layout()
        workers = []
        for path in sorted((self.directory / "steps").glob("**/_meta/inputs.json")):
            directory = path.parent.parent
            metadata = WorkerMetadata(directory)
            definition = metadata.calculation_set(metadata.inputs.read(), ".")
            worker = create_worker(definition["batches"][0]["runtime"], directory=directory)
            if isinstance(worker, SingleWorker):
                worker.wdir_name = "."
            worker._retrieve_mode = "all"
            workers.append(worker)
        return workers

    def reset(self):
        """Clear transient state; durable chain state belongs to its directory."""
        self.atoms = None

    def as_dict(self):
        return copy.deepcopy(self.runtime_config)

    def read_trajectory(self, add_step_info=False, archive_path=None, **kwargs):
        path = self.directory / self.TRAJ_NAME
        if archive_path is None:
            frames = read(path, ":") if path.exists() else []
        else:
            from io import StringIO
            from gdpx.utils.archive import open_archive
            frames = []
            with open_archive(archive_path) as archive:
                for member in archive:
                    if member.name == f"{self.directory.name}/{self.TRAJ_NAME}":
                        frames = read(StringIO(archive.extractfile(member).read().decode()), ":", format="extxyz")
                        break
        if add_step_info:
            for frame in frames:
                frame.info["step"] = frame.info.get("mc_step", 0)
        return frames
