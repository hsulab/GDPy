import copy
import enum
import shutil
from collections.abc import Mapping
from contextlib import nullcontext

import numpy as np
from ase import Atoms, data
from ase.io import read, write

from gdpx.core.output import quiet_logging
from gdpx.execution.workers.drive import DriverBasedWorker
from gdpx.execution.workers.single import SingleWorker

from ..exploration import BaseExploration
from ..move_step import read_pending, run_worker_move
from ..accepted_state import load_accepted_state, save_accepted_state
from ..checkpoint import load_data, save_data, publish_snapshot, read_snapshot, prune_snapshots
from ..sampling.moves.operator import BaseMCOperator
from ..sampling import parse_operators
from ..sampling.geometry import infer_unique_atomic_numbers, prepare_operators
from .output import mc_box, report_outcome, report_setup, report_status

"""This module tries to offer a base class for all MonteCarlo-like methods.
"""

MC_EARLYSTOP_FNAME = "MC_EARLY_STOPPED"

MCStepState = enum.Enum("MCStepState", ("UNFINISHED", "FINISHED", "FAILED", "EARLYSTOPPED"))


_EXCHANGE_OPERATORS = {
    "exchange", "biased_volume_exchange", "cavity_exchange", "adsorbate_exchange",
}


def create_monte_carlo(
    system, strategy, convergence=None, checkpoint=None, output=None,
    random_seed=None, directory="./",
):
    """Create standard MC from its public single-system configuration."""
    if not isinstance(system, Mapping):
        raise TypeError("monte_carlo system must be a mapping.")
    unknown = system.keys() - {"builder", "ensemble", "ignore_atoms_tags"}
    if unknown:
        raise ValueError(f"Unsupported monte_carlo system settings: {', '.join(sorted(unknown))}.")
    if "builder" not in system:
        raise ValueError("monte_carlo requires system.builder.")
    if not isinstance(system.get("ignore_atoms_tags", True), bool):
        raise TypeError("system.ignore_atoms_tags must be a boolean.")
    if not isinstance(strategy, Mapping):
        raise TypeError("monte_carlo strategy must be a mapping.")
    unknown = strategy.keys() - {"operators"}
    if unknown:
        raise ValueError(f"Unsupported monte_carlo strategy settings: {', '.join(sorted(unknown))}.")
    operators = strategy.get("operators")
    if not isinstance(operators, list) or not operators:
        raise ValueError("monte_carlo requires a nonempty strategy.operators list.")

    ensemble = system.get("ensemble")
    if not isinstance(ensemble, Mapping):
        raise ValueError("monte_carlo requires system.ensemble.")
    unknown = ensemble.keys() - {"method", "temperature", "chemical_potentials"}
    if unknown:
        raise ValueError(f"Unsupported ensemble settings: {', '.join(sorted(unknown))}.")
    ensemble_method = ensemble.get("method")
    allowed_ensembles = {"canonical", "semi_grand_canonical", "grand_canonical"}
    if ensemble_method not in allowed_ensembles:
        raise ValueError(
            "system.ensemble.method must be canonical, semi_grand_canonical, or grand_canonical."
        )
    temperature = ensemble.get("temperature")
    if not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("system.ensemble.temperature must be finite and positive.")
    chemical_potentials = ensemble.get("chemical_potentials", {})
    if not isinstance(chemical_potentials, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, (int, float)) or not np.isfinite(value)
        for name, value in chemical_potentials.items()
    ):
        raise ValueError("system.ensemble.chemical_potentials must map particle names to finite values.")
    if ensemble_method == "canonical" and chemical_potentials:
        raise ValueError("The canonical ensemble does not accept chemical_potentials.")
    if ensemble_method != "canonical" and not chemical_potentials:
        raise ValueError(f"The {ensemble_method} ensemble requires chemical_potentials.")

    resolved_operators = copy.deepcopy(operators)
    for operator in resolved_operators:
        if not isinstance(operator, Mapping):
            raise TypeError("Every strategy operator must be a mapping.")
        if "temperature" in operator or "chempots" in operator:
            raise ValueError(
                "Move operator temperature and chempots to system.ensemble; "
                "operators contain proposal settings only."
            )
        operator["temperature"] = float(temperature)
        name = operator.get("method", "move")
        particles = operator.get("particles", [])
        if name == "swap_type":
            if ensemble_method not in {"semi_grand_canonical", "grand_canonical"}:
                raise ValueError("swap_type requires a semi_grand_canonical or grand_canonical ensemble.")
            missing = [particle for particle in particles if particle not in chemical_potentials]
            if missing:
                raise ValueError(f"Missing chemical potentials for: {', '.join(missing)}.")
            operator["chempots"] = [chemical_potentials[particle] for particle in particles]
        elif name in _EXCHANGE_OPERATORS:
            if ensemble_method != "grand_canonical":
                raise ValueError(f"{name} requires a grand_canonical ensemble.")
            missing = [particle for particle in particles if particle not in chemical_potentials]
            if missing:
                raise ValueError(f"Missing chemical potentials for: {', '.join(missing)}.")
            operator["chempots"] = [chemical_potentials[particle] for particle in particles]
        elif name == "react":
            if ensemble_method != "grand_canonical":
                raise ValueError("react requires a grand_canonical ensemble.")
            reaction = operator.get("reaction", {})
            reaction_particles = reaction.get("particles", [])
            missing = [particle for particle in reaction_particles if particle not in chemical_potentials]
            if missing:
                raise ValueError(f"Missing chemical potentials for: {', '.join(missing)}.")
            if "chempot_0" in reaction:
                raise ValueError("Move reaction.chempot_0 to system.ensemble.chemical_potentials.")
            reaction["chempot_0"] = [chemical_potentials[particle] for particle in reaction_particles]

    checkpoint = {} if checkpoint is None else checkpoint
    output = {} if output is None else output
    if not isinstance(checkpoint, Mapping) or checkpoint.keys() - {"period"}:
        raise ValueError("checkpoint accepts only period.")
    if not isinstance(output, Mapping) or output.keys() - {"dump_period"}:
        raise ValueError("output accepts only dump_period.")
    ckpt_period = checkpoint.get("period", 100)
    dump_period = output.get("dump_period", 1)
    if not isinstance(ckpt_period, int) or ckpt_period <= 0:
        raise ValueError("checkpoint.period must be a positive integer.")
    if not isinstance(dump_period, int) or dump_period <= 0:
        raise ValueError("output.dump_period must be a positive integer.")
    if convergence is None:
        convergence = {"steps": 1}
    if not isinstance(convergence, Mapping):
        raise TypeError("monte_carlo convergence must be a mapping.")

    engine = MonteCarlo(
        builder=system["builder"], operators=resolved_operators,
        convergence=copy.deepcopy(dict(convergence)), random_seed=random_seed,
        dump_period=dump_period, ckpt_period=ckpt_period,
        ignore_atoms_tags=system.get("ignore_atoms_tags", True),
        should_retry=False, restart=False, directory=directory,
    )
    engine.system_config = {
        "ensemble": copy.deepcopy(dict(ensemble)),
        "ignore_atoms_tags": system.get("ignore_atoms_tags", True),
    }
    engine.strategy_config = copy.deepcopy(dict(strategy))
    return engine


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


class MonteCarlo(BaseExploration):
    restart = False
    requires_single_point_runtime = True

    #: Prefix of the working directory.
    WDIR_PREFIX: str = "cand"

    #: Name of the MC trajectory.
    TRAJ_NAME: str = "mc.xyz"

    #: Name of the file stores MC information (operations).
    INFO_NAME: str = "opstat.txt"

    def __init__(
        self,
        builder: dict,
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

        # Check system type
        self.register_builder(builder)

        # Parse operators
        self.operators, self.op_probs = parse_operators(operators)

        # Parse convergence
        self.convergence = convergence
        if self.convergence.get("steps", None) is None:
            self.convergence["steps"] = 1

        return

    def register_worker(self, worker, *args, **kwargs) -> None:
        """Attach an energy-only runtime for ordinary Monte Carlo."""
        candidate = worker[0] if isinstance(worker, list) and worker else worker
        if self.requires_single_point_runtime:
            method = getattr(getattr(getattr(candidate, "runtime", None), "config", None), "executor", None)
            method = getattr(method, "method", None)
            if method != "spc":
                raise ValueError(
                    "monte_carlo requires runtime.executor.method: spc so trial structures "
                    "are evaluated without relaxation or dynamics."
                )
        super().register_worker(worker, *args, **kwargs)

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
        self.worker.directory = self.directory / "calculations" / "step.0000"
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
        self.worker.wdir_name = "cand0"
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
            step_converged = True
        else:
            step_converged = False

        return step_converged

    def _attach_bond_length_minimum_list(self):
        """Find possible elements in the simulation and build a bond-distance list."""
        numbers = infer_unique_atomic_numbers(self.operators, substrates=[self.atoms])
        if hasattr(self.builder, "get_custom_pair_distance_dict"):
            raise ValueError("Monte Carlo does not support custom pair distances yet.")
        prepare_operators(self.operators, numbers)

        return

    def run(self, *args, **kwargs):
        """Run MonteCarlo simulation."""
        # Subclasses such as hybrid MC still own their procedure-level output.
        logging_context = quiet_logging() if type(self)._run is MonteCarlo._run else nullcontext()
        with logging_context:
            return self._run_with_worker(*args, **kwargs)

    def _run_with_worker(self, *args, **kwargs):
        self._validate_calculation_layout()
        super().run(*args, **kwargs)

        # Check if it has a valid worker
        if isinstance(self.worker, DriverBasedWorker):
            self._print("Convert a DriverBasedWorker to a SingleWorker.")
            self.worker = SingleWorker.from_a_worker(self.worker)
        assert isinstance(self.worker, SingleWorker), (
            f"{self.__class__.__name__} only supports SingleWorker (set use_single=True)."
        )
        self.worker.directory = self.directory

        # Create an atoms during the run-time
        # If it is created in init, it will be re-used in active-learning loop.
        # Thus, we create a new one every run time.
        frames = self.builder.run()
        assert len(frames) == 1, f"{self.__class__.__name__} only accepts one structure."
        self.atoms: Atoms = frames[0]

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
        report_setup(self.operators, self.op_probs, self.convergence["steps"], self.random_seed)

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
                    self._save_checkpoint(step=curr_step)
                    # -- clean up
                    if ((self.directory / "calculations" / f"step.{curr_step:04d}").exists()) and (
                        curr_step % self.dump_period != 0
                    ):
                        shutil.rmtree(self.directory / "calculations" / f"step.{curr_step:04d}")
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
        self.worker.directory = self.directory / "calculations" / f"step.{istep:04d}"
        self.worker.wdir_name = "cand0"
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
        write(self.directory / self.TRAJ_NAME, self.atoms, append=True)
        if (self.directory / f"pending-move-{istep}").exists():
            self._save_checkpoint(istep, force=True)
        return self._check_earlystop(self.atoms)

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
                  operators=[op.as_dict() for op in self.operators], rng=self.rng.bit_generator.state,
                  output_sizes={name: (self.directory / name).stat().st_size
                                for name in (self.TRAJ_NAME, "mc_attempts.xyz", self.INFO_NAME)
                                if (self.directory / name).exists()}))
        publish_snapshot(self.directory, staging, destination, "checkpoint.")
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

        # Reset mctraj
        self._print("Reset `mc.xyz` and `mc_attempts.xyz`.")
        mctraj = read(self.directory / self.TRAJ_NAME, f":{step + 1}")
        write(self.directory / self.TRAJ_NAME, mctraj)

        mctraj_attempts = read(self.directory / "mc_attempts.xyz", f":{step + 1}")
        write(self.directory / "mc_attempts.xyz", mctraj_attempts)

        # Reset opstat.txt
        self._print("Reset `opstat.txt`.")
        # We do not have the info for cand0 but have one comment line
        # so the final number of lines equal step+1
        with open(self.directory / self.INFO_NAME, "r") as fopen:
            opstat_lines = fopen.readlines()
        with open(self.directory / self.INFO_NAME, "w") as fopen:
            fopen.write("".join(opstat_lines[: step + 1]))

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
        """Check the convergence of MC.

        Currently, the only criteria is whether the simulation reaches the maximum
        steps.

        """
        converged = False
        if (self.directory / self.TRAJ_NAME).exists():
            mctraj = read(self.directory / self.TRAJ_NAME, ":")
            nframes = len(mctraj)
            # self.start_step = nframes
            if nframes > self.convergence["steps"]:
                converged = True
            if (self.directory / MC_EARLYSTOP_FNAME).exists():
                converged = True
        else:
            ...

        return converged

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
        if ((metadata.exists() and not scheduler_metadata) or (self.directory / "_data").exists()
                or any(self.directory.glob("_*_jobs.json")) or any(self.directory.glob("cand*"))
                or any(self.directory.glob("step.*")) or (self.directory / "init" / "_meta").exists()):
            raise ValueError("Legacy append-based exploration layout; use a new working directory.")
        for path in (self.directory / "calculations").glob("**/_meta"):
            WorkerMetadata(path.parent).compact

    def _prune_calculations(self, step):
        for path in (self.directory / "calculations").glob("step.*"):
            if int(path.name.split(".")[1]) > step:
                shutil.rmtree(path)

    def get_workers(self):
        """Reconstruct workers from each retained calculation's saved runtime."""
        from gdpx.execution.factory import create_worker
        from gdpx.execution.workers.metadata import WorkerMetadata

        self._validate_calculation_layout()
        workers = []
        for path in sorted((self.directory / "calculations").glob("**/_meta/inputs.json")):
            directory = path.parent.parent
            metadata = WorkerMetadata(directory)
            definition = metadata.calculation_set(metadata.inputs.read(), ".")
            worker = create_worker(definition["batches"][0]["runtime"], directory=directory)
            if isinstance(worker, SingleWorker):
                worker.wdir_name = "cand0"
            worker._retrieve_mode = "all"
            workers.append(worker)
        return workers

    def as_dict(self) -> dict:
        """"""
        potential = self.worker.runtime.provider_potential
        if hasattr(potential, "remove_loaded_models"):
            potential.remove_loaded_models()
        if hasattr(self, "system_config"):
            system = copy.deepcopy(self.system_config)
            system["builder"] = self.builder.as_dict()
            return copy.deepcopy({
                "method": "monte_carlo",
                "random_seed": self.random_seed,
                "system": system,
                "strategy": self.strategy_config,
                "convergence": self.convergence,
                "checkpoint": {"period": self.ckpt_period},
                "output": {"dump_period": self.dump_period},
                "runtime": self.worker.as_dict(),
            })
        operators = [op.as_dict() for op in self.operators]
        recipe = {
            "random_seed": self.random_seed,
            "builder": self.builder.as_dict(),
            "operators": operators,
            "convergence": self.convergence,
            "dump_period": self.dump_period,
            "ckpt_period": self.ckpt_period,
            "ignore_atoms_tags": self.ignore_atoms_tags,
            "should_retry": self.should_retry,
            "restart": self.restart,
        }
        engine_params = {
            "method": "monte_carlo",
            "recipe": recipe,
            "runtime": self.worker.as_dict(),
        }
        return copy.deepcopy(engine_params)
