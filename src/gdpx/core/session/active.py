#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
import time
from typing import Tuple, Union

from ..operation import Operation
from .session import AbstractSession, SessionState
from .utils import traverse_postorder


def set_node_directory_in_active_session(
    node: Operation, node_index, working_directory: pathlib.Path
) -> None:
    """"""
    prev_name = node.directory.name.split(".")[-1]  # remove previous orders
    if not prev_name:
        prev_name = node.__class__.__name__
    node.directory = working_directory / f"{node_index:>04d}.{prev_name}"

    return


class ActiveSession(AbstractSession):

    def __init__(
        self,
        steps: int = 2,
        reset_random_state: bool = False,
        reset_random_config: Tuple[str, int] = ("init", 0),
        directory: Union[str, pathlib.Path] = "./",
    ) -> None:
        """Initialise an ActiveSession.

        Args:
            steps: Number of active learning steps.
            reset_random_seed: A tuple of a str and a int.

        """
        self.steps = steps

        # Some random-related parameters
        self.reset_random_state = reset_random_state
        assert reset_random_config[0] in [
            "init",
            "zero",
        ], "Reset random seed mode must either be init or zero."
        self.reset_random_seed_mode = reset_random_config[0]
        self.reset_random_seed_step = reset_random_config[1]

        self.directory = pathlib.Path(directory)

        return

    def run(self, operation: Operation, feed_dict: dict = {}) -> None:
        """"""
        self.state = SessionState.StepToStart
        # Update nodes' attrs based on the previous iteration
        # nodes_postorder = traverse_postorder(operation)
        # for node in nodes_postorder:
        #    if hasattr(node, "enable_active"):
        #        node.enable_active()
        if self.reset_random_state:
            self._print(
                f"RESET RANDOM SEED - MODE: {self.reset_random_seed_mode} STEP: {self.reset_random_seed_step}"
            )

        # Run iterative steps
        for istep in range(self.steps):
            curr_wdir = self.directory / f"iter.{istep:>04d}"
            # Find forward order
            nodes_postorder = traverse_postorder(operation)

            # Check random states
            if (
                self.reset_random_state
                and istep >= self.reset_random_seed_step
            ):
                for node in nodes_postorder:
                    if hasattr(node, "reset_random_seed"):
                        self._print(
                            f"reset {node.directory.name}'s random seeds."
                        )
                        node.reset_random_seed(
                            mode=self.reset_random_seed_mode
                        )

            # Run operations
            self._run_nodes(
                curr_wdir,
                nodes_postorder=nodes_postorder,
                feed_dict=feed_dict,
                reset_states=True,
                set_node_dir_func=set_node_directory_in_active_session,
            )

            # Check state
            if not (self.state == SessionState.StepFinished):
                self._print("wait current iteration to finish...")
            else:
                # If previous step finished, the nodes may not have outputs
                # as we skip them...
                if not (curr_wdir / "FINISHED").exists():
                    # Report convergence
                    self._print(f"[{'CONVERGENCE':^24s}]")
                    converged_list = []
                    for node in nodes_postorder:
                        if hasattr(node, "report_convergence"):
                            converged = node.report_convergence()
                            converged_list.append(converged)
                    if converged_list and all(converged_list):
                        self._print(
                            f"Active Session converged at step {istep}."
                        )
                        self.state = SessionState.LoopConverged
                    else:
                        self._print(
                            f"Active Session UNconverged at step {istep}."
                        )
                        if istep + 1 == self.steps:
                            self.state = SessionState.LoopUnConverged
                        else:
                            ...  # Just StepFinished
                    # Save state to a file
                    with open(curr_wdir / "FINISHED", "w") as fopen:
                        fopen.write(
                            f"STATE {self.state} FINISHED AT {time.asctime( time.localtime(time.time()) )}."
                        )
                else:
                    self._print(
                        "[{:^24s}] FINISHED".format(
                            f"STEP.{str(istep).zfill(4)}"
                        )
                    )
            # Add an atrribute that indicates all steps are finished
            if self.state != SessionState.StepFinished:
                break
        else:
            self.state = SessionState.LoopFinished

        return


if __name__ == "__main__":
    ...
