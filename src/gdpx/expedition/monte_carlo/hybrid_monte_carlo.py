#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools

from ase.io import read, write

from .. import DriverBasedWorker, SingleWorker, dict2str, registers
from ..expedition import parse_worker
from .monte_carlo import MonteCarlo
from .operators import select_operator

MC_EARLYSTOP_FNAME = "MC_EARLY_STOPPED"
EARLYSTOP_STATE_NAME = "EARLYSTOPPED"


class HybridMonteCarlo(MonteCarlo):

    def __init__(self, procedure, extra_workers={}, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.procedure = procedure
        self.extra_workers = extra_workers

        return

    def _run(self, *args, **kwargs):
        """"""
        # set init worker
        self.worker.directory = self.directory / "init"

        # check if subprocedures in the procedure are all valid
        procedure_steps = []
        for subprocedure in self.procedure:
            if isinstance(subprocedure, list):
                assert len(subprocedure) == 2 and subprocedure[0] == "monte_carlo", ""
                worker_name = subprocedure[1].split("_")[1]
                worker_params = self.extra_workers.get(worker_name, None)
                if worker_params is not None:
                    subworker = parse_worker(worker_params)
                    if isinstance(subworker, DriverBasedWorker):
                        self._print("Convert a DriverBasedWorker to a SingleWorker.")
                        subworker = SingleWorker.from_a_worker(subworker)
                    assert isinstance(
                        subworker, SingleWorker
                    ), f"{self.__class__.__name__} only supports SingleWorker (set use_single=True) but {subprocedure} is not."
                    subworker.directory = self.directory / "mc"
                    subproc_func = functools.partial(self._irun_metropolis, worker=subworker)
                    procedure_steps.append(("mc", subproc_func))
                else:
                    raise RuntimeError(f"Unknown subprocedure with worker {subprocedure}.")
            elif subprocedure.startswith("worker"):
                worker_name = subprocedure.split("_")[1]
                worker_params = self.extra_workers.get(worker_name, None)
                if worker_params is not None:
                    subworker = parse_worker(worker_params)
                    if isinstance(subworker, DriverBasedWorker):
                        self._print("Convert a DriverBasedWorker to a SingleWorker.")
                        subworker = SingleWorker.from_a_worker(subworker)
                    assert isinstance(
                        subworker, SingleWorker
                    ), f"{self.__class__.__name__} only supports SingleWorker (set use_single=True) but {subprocedure} is not."
                    subworker.directory = self.directory / worker_name
                    subproc_func = functools.partial(self._irun_dynamics, worker=subworker)
                    procedure_steps.append((worker_name, subproc_func))
                else:
                    raise RuntimeError(f"Unknown subprocedure with worker {subprocedure}.")
            else:
                raise RuntimeError(f"Unknown subprocedure {subprocedure}.")

        # enter the main loop
        converged = self.read_convergence()
        if not converged:
            # init structure
            step_converged = False
            if not self._verify_checkpoint():
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

                step_state = "UNFINISHED"
                for subproc_name, subproc_func in procedure_steps:
                    step_state = subproc_func(name=subproc_name, step=curr_step)
                    if step_state == "UNFINISHED":
                        self._print("Wait MC step to finish.")
                        break
                    elif step_state == "FINISHED":
                        # post worker compute with MC
                        ...
                    elif step_state == "FAILED":
                        self._print(f"RETRY STEP {curr_step}.")
                        break
                    elif step_state == EARLYSTOP_STATE_NAME:
                        # We need a file flag to indicate the simutlation is finshed
                        # when read_convergence is called.
                        with open(self.directory / MC_EARLYSTOP_FNAME, "w") as fopen:
                            fopen.write(f"{step_state =}")
                        break
                    else:
                        ...
                else:
                    curr_step += 1
                if step_state != "FINISHED":
                    break
        else:
            self._print("Monte Carlo is converged.")

        return

    def _irun_dynamics(self, step: int, name: str, worker: SingleWorker):
        """"""
        self._print(f"===== MC Step {step} {name.upper()} =====")
        worker.wdir_name = f"{self.WDIR_PREFIX}{step}"

        curr_atoms = self.atoms
        curr_tags = curr_atoms.get_tags()

        _ = worker.run([curr_atoms])
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            curr_atoms = worker.retrieve()[0][-1]
            curr_atoms.set_tags(curr_tags)

            self.energy_operated = curr_atoms.get_potential_energy()
            self._print(f"post ene: {self.energy_operated}")

            self.energy_stored = self.energy_operated
            self.atoms = curr_atoms

            step_state = self._check_earlystop(self.atoms)
        else:
            step_state = "UNFINISHED"

        return step_state

    def _irun_metropolis(self, step: int, name: str, worker: SingleWorker) -> str:
        """Run a single MC step.

        Each step has three status as FINISHED, UNFINISHED, and FAILED.

        """
        self._print(f"===== MC Step {step} {name.upper()} =====")
        self._print(f"RANDOM_SEED:  {self.random_seed}")
        for l in dict2str(self.rng.bit_generator.state).split("\n"):
            self._print(l)

        worker.wdir_name = f"{self.WDIR_PREFIX}{step}"

        # - operate atoms
        curr_op = select_operator(self.operators, self.op_probs, self.rng)
        self._print(f"operator {curr_op.__class__.__name__}")
        curr_atoms = curr_op.run(self.atoms, self.rng)
        if curr_atoms:  # is not None
            # --- add info
            curr_atoms.info["confid"] = int(f"{step}")
            curr_atoms.info["step"] = -1  # NOTE: remove step info from driver
        else:
            self._print("FAILED to run operation...")

        # - run postprocess
        if curr_atoms is not None:
            # - TODO: save some info not stored by driver
            curr_tags = curr_atoms.get_tags()

            # - run postprocess (spc, min or md)
            _ = worker.run([curr_atoms], read_ckpt=True)
            worker.inspect(resubmit=True)
            if worker.get_number_of_running_jobs() == 0:
                curr_atoms = worker.retrieve()[0][-1]
                curr_atoms.set_tags(curr_tags)

                self.energy_operated = curr_atoms.get_potential_energy()
                self._print(f"post ene: {self.energy_operated}")

                # -- metropolis
                success = curr_op.metropolis(
                    self.energy_stored, self.energy_operated, self.rng
                )

                self._save_step_info(curr_op, success)

                # -- update atoms
                if success:
                    self.energy_stored = self.energy_operated
                    self.atoms = curr_atoms
                    self._print("success...")
                else:
                    self._print("failure...")

                # FIXME: Save unaccepted structures as well?
                write(self.directory / self.TRAJ_NAME, self.atoms, append=True)

                # -- check earlystopping
                # We earlystop the simulation at the end of each step and use
                # the MC-updated atoms, which may be unaccepted (failure) and
                # further lead the inconsistency in the final saved structure.
                # After several tests, it is better to check on accepted structures
                # so ignore the below comment
                step_state = self._check_earlystop(self.atoms)

            else:
                step_state = "UNFINISHED"
        else:
            # save the previous structure as the current operation gives no structure.
            step_state = "FAILED"

        return step_state


if __name__ == "__main__":
    ...
