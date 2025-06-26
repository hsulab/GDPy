#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools

from ase.io import write

from gdpx.factory.computer import canonicalise_worker
from gdpx.utils.strconv import dictionary_to_string
from gdpx.worker.drive import DriverBasedWorker
from gdpx.worker.single import SingleWorker

from .monte_carlo import MCStepState, MonteCarlo
from .operators import select_operator

MC_EARLYSTOP_FNAME = "MC_EARLY_STOPPED"


class HybridMonteCarlo(MonteCarlo):

    def __init__(self, procedure, num_mcmoves: int, extra_workers={}, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.procedure = procedure

        self.num_mcmoves = num_mcmoves

        self.extra_workers = extra_workers

        return

    def _run(self, *args, **kwargs):
        """"""
        # set init worker
        self.worker.directory = self.directory / "init"

        # Format indent
        for op in self.operators:
            op.indent = "  "

        # check if subprocedures in the procedure are all valid
        procedure_steps = []
        for subprocedure in self.procedure:
            if isinstance(subprocedure, list):
                assert len(subprocedure) == 2 and subprocedure[0] == "monte_carlo", ""
                worker_name = subprocedure[1].split("_")[1]
                worker_params = self.extra_workers.get(worker_name, None)
                if worker_params is not None:
                    subworker = canonicalise_worker(worker_params)
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
                    subworker = canonicalise_worker(worker_params)
                    # if isinstance(subworker, DriverBasedWorker):
                    #     self._print("Convert a DriverBasedWorker to a SingleWorker.")
                    #     subworker = SingleWorker.from_a_worker(subworker)
                    # assert isinstance(
                    #     subworker, SingleWorker
                    # ), f"{self.__class__.__name__} only supports SingleWorker (set use_single=True) but {subprocedure} is not."
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

                step_state = MCStepState.UNFINISHED
                self._print(f"===== Hybrid MC Step {curr_step} =====")
                for subproc_name, subproc_func in procedure_steps:  # [dynamics, mcmove]
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
                    curr_step += 1
                if step_state != MCStepState.FINISHED:
                    break
        else:
            self._print("Monte Carlo is converged.")

        return

    def _irun_dynamics(self, step: int, name: str, worker: DriverBasedWorker) -> MCStepState:
        """"""
        self._print(f">>>>> {name.upper()} ")
        worker.directory = self.directory / f"step.{step:>04d}" / "excurs"

        # Get tags as it is not stored by the worker.
        curr_atoms = self.atoms
        curr_tags = curr_atoms.get_tags()

        _ = worker.run([curr_atoms])
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            curr_atoms = worker.retrieve()[0][-1]
            curr_atoms.set_tags(curr_tags)

            self.energy_operated = curr_atoms.get_potential_energy()
            self._print(f"  ene {self.energy_stored:>18.4f} -> {self.energy_operated:>18.4f}")

            self.energy_stored = self.energy_operated
            self.atoms = curr_atoms

            step_state = self._check_earlystop(self.atoms)
        else:
            step_state = MCStepState.UNFINISHED

        return step_state

    def _irun_metropolis(self, step: int, name: str, worker: SingleWorker) -> MCStepState:
        """Run a single MC step.

        Each step has three status as FINISHED, UNFINISHED, and FAILED.

        """
        self._print(f">>>>> {name.upper()} ")
        self._print(f"RANDOM_SEED:  {self.random_seed}")
        for l in dictionary_to_string(self.rng.bit_generator.state).split("\n"):
            self._print(l)

        worker.directory = self.directory / f"step.{step:>04d}" / "mcmove"

        # TODO: Maybe we can group all spcs into one job by a socket-based calculator
        #       if one spc is expensive, for example, a DFT calculation.
        for i in range(self.num_mcmoves):
            # Update worker calculation folder name
            worker.wdir_name = f"{self.WDIR_PREFIX}{i}"

            # Run mcmove
            curr_op = select_operator(self.operators, self.op_probs, self.rng)

            self._print(f"  >>> mcmove.{i:>04d}  {curr_op.name} ")
            curr_atoms = curr_op.run(self.atoms, self.rng)
            if curr_atoms:  # is not None
                # Add info to atoms and remove step info from driver
                curr_atoms.info["confid"] = int(f"{step}")
                curr_atoms.info["step"] = -1
            else:
                self._print(
                    "  FAILED to run operation..."
                )  # Due to absence of particles in the region or neighbour distance restraints

            # Run single-point-calculation and metropolis
            if curr_atoms is not None:
                # Save tags
                curr_tags = curr_atoms.get_tags()

                # single-point calculation
                _ = worker.run([curr_atoms], read_ckpt=True)
                worker.inspect(resubmit=True)
                if worker.get_number_of_running_jobs() == 0:
                    curr_atoms = worker.retrieve()[0][-1]
                    curr_atoms.set_tags(curr_tags)

                    self.energy_operated = curr_atoms.get_potential_energy()
                    self._print(f"  ene {self.energy_stored:>18.4f} -> {self.energy_operated:>18.4f}")

                    # run metropolis
                    success = curr_op.metropolis(self.energy_stored, self.energy_operated, self.rng)
                    self._save_step_info(curr_op, success)  # TODO: save step info in step folder only

                    if success:
                        self.energy_stored = self.energy_operated
                        self.atoms = curr_atoms
                        self._print("  <<< success")
                    else:
                        self._print("  <<< failure")

                    # check earlystopping
                    step_state = self._check_earlystop(self.atoms)
                else:
                    step_state = MCStepState.UNFINISHED
                    break
            else:
                # save the previous structure as the current operation gives no structure.
                step_state = MCStepState.FAILED
        else:
            ...  # If we reach here, all mcmoves are finished

        # Save the final structure only
        write(self.directory / self.TRAJ_NAME, self.atoms, append=True)

        return step_state


if __name__ == "__main__":
    ...
