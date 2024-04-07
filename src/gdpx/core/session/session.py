#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Callable

from .. import config
from ..operation import Operation


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
        assert state in SESSION_STATE_LIST, f"Invalid state `{state}` to assign."
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
                is_broken = any(broken_states)
                if not is_broken:
                    self.state = "StepToContinue"
                    self._print("  wait previous nodes to finish...")
                else:
                    # The `broken` status is contagious
                    node.status = "exit"
                    self.state = "StepBroken"
                    self._print("  The current node is broken.")
        else:
            self.state = "StepBroken"
            self._print("  The current node is broken.")

        return


if __name__ == "__main__":
    ...
