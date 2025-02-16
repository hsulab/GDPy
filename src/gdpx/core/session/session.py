#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Callable, List, Optional

from .. import config
from ..operation import Operation
from ..placeholder import Placeholder
from ..variable import Variable

#: A List of valid session states.
SESSION_STATE_LIST: List[str] = [
    "StepToStart",
    "StepFinished",
    "StepToContinue",
    "StepBroken",
    "LoopToStart",
    "LoopFinished",
    "LoopConverged",
    "LoopUnConverged",
]

#: A List of finished session states.
FINISHED_SESSION_STATES: List[str] = [
    "StepBroken",
    "LoopFinished",
    "LoopConverged",
    "LoopUnConverged",
]


class AbstractSession:

    #: Session State.
    _state: str = "LoopToStart"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    @property
    def state(self) -> str:
        """"""

        return self._state

    @state.setter
    def state(self, state: str):
        """"""
        assert (
            state in SESSION_STATE_LIST
        ), f"Invalid state `{state}` to assign."
        self._state = state

        return

    def is_finished(self) -> bool:
        """"""
        is_finished = False
        if self.state in FINISHED_SESSION_STATES:
            is_finished = True

        return is_finished

    def _process_operation(self, node: Operation):
        """"""
        if not node.is_about_to_exit():
            if node.is_ready_to_forward():  # All input nodes finished.
                node.inputs = [
                    input_node.output for input_node in node.input_nodes
                ]
                node.output = node.forward(*node.inputs)
            else:
                # - check whether this node' not ready due to previous nodes are broken.
                broken_states = []
                for input_node in node.input_nodes:
                    if isinstance(input_node, Operation):
                        broken_states.append(input_node.is_about_to_exit())
                    else:
                        broken_states.append(False)
                self._debug(f"{node.input_nodes =}")
                self._debug(f"{broken_states =}")
                is_broken = any(broken_states)
                if not is_broken:
                    self.state = "StepToContinue"
                    self._print(
                        "\x1b[1;33;40m"
                        + "  wait previous nodes to finish..."
                        + "\x1b[0m"
                    )
                else:
                    # The `broken` status is contagious
                    node.status = "exit"
                    self.state = "StepBroken"
                    self._print(
                        "\x1b[1;31;40m"
                        + "  The current node is broken."
                        + "\x1b[0m"
                    )
        else:
            self.state = "StepBroken"
            self._print(
                "\x1b[1;31;40m" + "  The current node is broken." + "\x1b[0m"
            )

        return

    def _run_nodes(
        self,
        wdir: pathlib.Path,
        nodes_postorder: list[Operation],
        feed_dict: dict = {},
        reset_states: bool = False,
        set_node_dir_func: Optional[Callable] = None,
    ):
        """"""
        assert set_node_dir_func is not None

        if (wdir / "FINISHED").exists():
            self.state = "StepFinished"
            return

        # Clear previous nodes' outputs somtimes two steps run consecutively
        # and some nodes in the second step breaks and make its following nodes # use outputs from the last step, which is difficult to debug.
        if reset_states:
            for node in nodes_postorder:
                node.reset()

        # Show session information
        num_nodes = len(nodes_postorder)
        self._print(
            "\x1b[1;34;40m"
            + f"[{'START':^24s}] NUM_NODES: {num_nodes} AT MAIN: "
            + "\x1b[0m"
        )
        self._print("\x1b[1;34;40m" + f"    {str(wdir)}" + "\x1b[0m")

        # Run nodes
        self.state = "StepFinished"
        for i, node in enumerate(nodes_postorder):
            # Update node version
            if hasattr(node, "version"):
                node.version = wdir.name

            # Reset directory since it maybe changed
            set_node_dir_func(node, i, wdir)
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
                assert isinstance(
                    node, Operation
                ), f"Unknown node type: {type(node)}"
                self._debug(f"node: {node}")
                self._process_operation(node)

        return


if __name__ == "__main__":
    ...
