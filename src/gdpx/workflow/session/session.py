#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import enum
import pathlib
from typing import Callable, Optional

from gdpx import config

from .operation import Operation
from .variable import Variable
from .placeholder import Placeholder
from .node import NodeKind, WorkflowNode


class SessionState(enum.Enum):
    """Represents different session states."""

    # The iteration is about to start.
    StepToStart = enum.auto()

    # The iteration is not finished yet.
    StepToContinue = enum.auto()

    # The iteration is finished.
    StepFinished = enum.auto()

    # The iteration is broken.
    StepBroken = enum.auto()

    # The session iterations are all finsihed.
    LoopFinished = enum.auto()

    # The session is converged at an iteration.
    LoopConverged = enum.auto()

    # The session is not converged until the last iteration.
    LoopUnConverged = enum.auto()

    def is_finished(self) -> bool:
        """"""
        is_finished = False
        if self in FINISHED_SESSION_STATES:
            is_finished = True

        return is_finished


FINISHED_SESSION_STATES: tuple[SessionState, ...] = (
    SessionState.StepBroken,
    SessionState.LoopFinished,
    SessionState.LoopConverged,
    SessionState.LoopUnConverged,
)


class BaseSession:

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    def __init__(self) -> None:
        """"""
        self._state = SessionState.StepToStart

        return

    @property
    def state(self) -> SessionState:
        """"""

        return self._state

    @state.setter
    def state(self, state: SessionState):
        """"""
        assert state in SessionState, f"Invalid state `{state}` to assign."
        self._state = state

        return

    def is_finished(self) -> bool:
        """"""

        return self.state.is_finished()

    def _process_operation(self, node: Operation) -> Optional[SessionState]:
        """"""
        state = None
        if not node.is_about_to_exit():
            if node.is_ready_to_forward():  # All input nodes finished.
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.forward(*node.inputs)
                if node.status == "unfinished":
                    state = SessionState.StepToContinue
                    self._print("\x1b[1;33;40m" + "  wait current node to finish..." + "\x1b[0m")
            else:
                # Check whether this node' not ready due to previous nodes are broken.
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
                    state = SessionState.StepToContinue
                    self._print("\x1b[1;33;40m" + "  wait previous nodes to finish..." + "\x1b[0m")
                else:
                    # The `broken` status is contagious
                    node.status = "exit"
                    state = SessionState.StepBroken
                    self._print("\x1b[1;31;40m" + "  The current node is broken." + "\x1b[0m")
        else:
            state = SessionState.StepBroken
            self._print("\x1b[1;31;40m" + "  The current node is broken." + "\x1b[0m")

        return state

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
            self.state = SessionState.StepFinished
            return

        # Clear previous nodes' outputs somtimes two steps run consecutively
        # and some nodes in the second step breaks and make its following nodes # use outputs from the last step, which is difficult to debug.
        if reset_states:
            for node in nodes_postorder:
                node.reset()

        # Show session information
        num_nodes = len(nodes_postorder)
        self._print("\x1b[1;34;40m" + f"[{'START':^24s}] NUM_NODES: {num_nodes} AT MAIN: " + "\x1b[0m")
        self._print("\x1b[1;34;40m" + f"    {str(wdir)}" + "\x1b[0m")

        # Run nodes
        self.state = SessionState.StepFinished
        for i, node in enumerate(nodes_postorder):
            # Update node version
            if hasattr(node, "version"):
                node.version = wdir.name

            # Reset directory since it maybe changed
            set_node_dir_func(node, i, wdir)
            if not isinstance(node, WorkflowNode):
                raise TypeError(f"Unknown workflow node: {type(node)}")
            node_type = node.node_kind.value
            self._print(
                "[{:^24s}] NAME: {} AT {}".format(
                    node_type,
                    node.__class__.__name__.upper(),
                    node.directory.name,
                )
            )

            if node.node_kind is NodeKind.PLACEHOLDER:
                node.output = feed_dict[node]
            elif node.node_kind is NodeKind.VARIABLE:
                node.output = node.value
            elif node.node_kind is NodeKind.OPERATION:
                if not isinstance(node, Operation):
                    raise TypeError(f"Operation node does not implement Operation: {type(node)}")
                self._debug(f"node: {node}")
                _state = self._process_operation(node)
                if _state is not None:
                    self.state = _state
            else:
                raise TypeError(f"Unknown node kind: {node.node_kind}")

        return


if __name__ == "__main__":
    ...
