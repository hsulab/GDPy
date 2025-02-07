#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
import time
from typing import Tuple, Union

from ..operation import Operation
from ..placeholder import Placeholder
from ..variable import Variable
from .session import AbstractSession
from .utils import traverse_postorder


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

    def run(
        self, operation: Operation, feed_dict: dict = {}, *args, **kwargs
    ) -> None:
        """"""
        self.state = "StepToStart"
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
        for curr_step in range(self.steps):
            curr_wdir = self.directory / f"iter.{str(curr_step).zfill(4)}"
            # -- run operation
            nodes_postorder = traverse_postorder(operation)
            if (
                self.reset_random_state
                and curr_step >= self.reset_random_seed_step
            ):
                for node in nodes_postorder:
                    if hasattr(node, "reset_random_seed"):
                        self._print(
                            f"reset {node.directory.name}'s random seeds."
                        )
                        node.reset_random_seed(
                            mode=self.reset_random_seed_mode
                        )
            # -- run operations
            self._irun(
                wdir=curr_wdir,
                nodes_postorder=nodes_postorder,
                feed_dict=feed_dict,
            )
            if not (self.state == "StepFinished"):
                self._print("wait current iteration to finish...")
            else:
                # If previous step finished, the nodes may not have outputs
                # as we skip them...
                if not (curr_wdir / "FINISHED").exists():
                    # Report convergence
                    self._print("[{:^24s}]".format("CONVERGENCE"))
                    converged_list = []
                    for node in nodes_postorder:
                        if hasattr(node, "report_convergence"):
                            converged = node.report_convergence()
                            converged_list.append(converged)
                    if converged_list and all(converged_list):
                        self._print(
                            f"Active Session converged at step {curr_step}."
                        )
                        self.state = "LoopConverged"
                    else:
                        self._print(
                            f"Active Session UNconverged at step {curr_step}."
                        )
                        if curr_step + 1 == self.steps:
                            self.state = "LoopUnConverged"
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
                            f"STEP.{str(curr_step).zfill(4)}"
                        )
                    )
            # Add an atrribute that indicates all steps are finished
            if self.state != "StepFinished":
                break
        else:
            self.state = "LoopFinished"

        return

    def _irun(
        self,
        wdir: Union[str, pathlib.Path],
        nodes_postorder: list[Operation],
        feed_dict: dict = {},
    ) -> None:
        """"""
        if (wdir / "FINISHED").exists():
            self.state = "StepFinished"
            return

        # Clear previous nodes' outputs somtimes two steps run consecutively
        # and some nodes in the second step
        # breaks and make its following nodes use outputs from the last step,
        # which is a hidden error
        for node in nodes_postorder:
            node.reset()

        # Find forward order
        self._print(
            "\x1b[1;34;40m"
            + "[{:^24s}] NUM_NODES: {} AT MAIN: ".format(
                "START", len(nodes_postorder)
            )
            + "\x1b[0m"
        )
        self._print("\x1b[1;34;40m" + "    {}".format(str(wdir)) + "\x1b[0m")

        # Run nodes
        self.state = "StepFinished"
        for i, node in enumerate(nodes_postorder):
            # -- change version ...
            if hasattr(node, "version"):
                node.version = wdir.name

            # Reset directory since it maybe changed
            prev_name = node.directory.name.split(".")[
                -1
            ]  # remove previous orders
            if not prev_name:
                prev_name = node.__class__.__name__
            # prev_name = node.__class__.__name__
            node.directory = wdir / f"{str(i).zfill(4)}.{prev_name}"
            if node.__class__.__name__.endswith("Variable"):
                node_type = "VX"
            else:
                node_type = "OP"
            self._print(
                "[{:^24s}] NAME: {} AT {}".format(
                    node_type,
                    node.__class__.__name__.upper(),
                    node.directory.name,
                )
            )

            if isinstance(node, Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else:  # Operation
                # FIXME: If the session has many branches,
                #        how do we define the state?
                assert isinstance(
                    node, Operation
                ), f"Unknown node type: {type(node)}"
                self._debug(f"node: {node}")
                self._process_operation(node)

        return


if __name__ == "__main__":
    ...
