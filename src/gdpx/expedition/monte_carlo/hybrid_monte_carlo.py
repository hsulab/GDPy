#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .. import DriverBasedWorker, SingleWorker, dict2str, registers
from .monte_carlo import MonteCarlo

MC_EARLYSTOP_FNAME = "MC_EARLY_STOPPED"
EARLYSTOP_STATE_NAME = "EARLYSTOPPED"


class HybridMonteCarlo(MonteCarlo):

    def __init__(self, host_worker: dict, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.register_worker(host_worker)
        self.host_worker = self.worker

        return

    def _run(self, *args, **kwargs):
        """"""
        # Besides the `self.worker`, we only need a host one.
        self._print(f"{self.worker =}")
        self.worker.directory = self.directory / "post"

        self._print(f"{self.host_worker =}")
        if isinstance(self.host_worker, DriverBasedWorker):
            self._print("Convert a DriverBasedWorker to a SingleWorker.")
            self.host_worker = SingleWorker.from_a_worker(self.host_worker)
        assert isinstance(
            self.host_worker, SingleWorker
        ), f"{self.__class__.__name__} only supports SingleWorker (set use_single=True)."
        self.host_worker.directory = self.directory / "host"

        # enter the main loop
        converged = self.read_convergence()
        if not converged:
            # init structure
            step_converged = False
            if not self._veri_checkpoint():
                step_converged = self._init_structure()
            else:
                step_converged = True
                self._load_checkpoint()

            if not step_converged:
                self._print("Wait structure to initialise.")
                return
            else:
                self.start_step += 1

            # run host + MC
            curr_step = self.start_step
            while True:
                if curr_step > self.convergence["steps"]:
                    self._print("Monte Carlo reaches the maximum step.")
                    break
                if (self.directory / MC_EARLYSTOP_FNAME).exists():
                    self._print("Monte Carlo reaches the earlystopping.")
                    break

                # host worker compute
                step_state = self._irun_host(curr_step)
                if step_state == "UNFINISHED":
                    self._print("Wait MC step to finish.")
                    break
                elif step_state == "FINISHED":
                    # post worker compute with MC
                    curr_step += 1
                elif step_state == "FAILED":
                    self._print(f"RETRY STEP {curr_step}.")
                elif step_state == EARLYSTOP_STATE_NAME:
                    # We need a file flag to indicate the simutlation is finshed
                    # when read_convergence is called.
                    with open(self.directory / MC_EARLYSTOP_FNAME, "w") as fopen:
                        fopen.write(f"{step_state =}")
                else:
                    ...
        else:
            self._print("Monte Carlo is converged.")

        return

    def _irun_host(self, step: int):
        """"""
        self._print(f"===== MC HOST Step {step} =====")
        worker = self.host_worker

        worker.wdir_name = f"cand{step}"

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


if __name__ == "__main__":
    ...
