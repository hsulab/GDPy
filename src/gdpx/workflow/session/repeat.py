"""Transactional repeated workflow execution."""

from __future__ import annotations

import pathlib
import time

import yaml

from gdpx.workflow.compiler import compile_workflow
from gdpx.workflow.configuration import OutputReference, WorkflowSpec
from gdpx.workflow.state import select_output
from gdpx.workflow.state_store import WorkflowStateStore

from .session import BaseSession, SessionState
from .utils import traverse_postorder


class RepeatSession(BaseSession):
    """Recompile each iteration from the last atomically committed state."""

    def __init__(self, spec: WorkflowSpec, directory: str | pathlib.Path = ".") -> None:
        self.spec = spec
        self.directory = pathlib.Path(directory)
        self.store = WorkflowStateStore(self.directory, spec)
        self.state = SessionState.StepToStart

    def _next_values(self, compiled) -> dict:
        values = {}
        for name, definition in self.spec.state.items():
            reference = definition.update
            node_name = reference.node if isinstance(reference, OutputReference) else reference
            node = compiled.nodes[node_name]
            if not hasattr(node, "output"):
                raise RuntimeError(f"State updater {node_name!r} produced no output.")
            value = node.output
            if isinstance(reference, OutputReference):
                value = select_output(value, reference.output)
            values[name] = value
        return values

    @staticmethod
    def _converged(nodes) -> bool:
        reports = [
            node.report_convergence()
            for node in nodes
            if hasattr(node, "report_convergence")
        ]
        return bool(reports) and all(reports)

    def _bind_state_artifacts(self, compiled, iteration: int) -> None:
        for state_name, definition in self.spec.state.items():
            reference = definition.update
            node_name = reference.node if isinstance(reference, OutputReference) else reference
            output = reference.output if isinstance(reference, OutputReference) else "default"
            node = compiled.nodes[node_name]
            if hasattr(node, "bind_state_artifact"):
                node.bind_state_artifact(output, state_name, self.directory, iteration)

    def run(self) -> None:
        last_iteration, state_values = self.store.load()
        if self.store.current.exists():
            manifest = yaml.safe_load(self.store.current.read_text(encoding="utf-8"))
            if manifest.get("converged", False):
                self.state = SessionState.LoopConverged
                return
        if last_iteration + 1 >= self.spec.settings.max_iterations:
            self.state = SessionState.LoopFinished
            return

        for iteration in range(last_iteration + 1, self.spec.settings.max_iterations):
            iteration_directory = self.directory / f"iter.{iteration:04d}"
            if (iteration_directory / "FINISHED").exists():
                raise RuntimeError(
                    f"Iteration {iteration} is marked finished without committed state; "
                    "use a fresh run directory."
                )
            compiled = compile_workflow(
                self.spec,
                iteration_directory,
                state_values=state_values if last_iteration >= 0 or state_values else None,
            )
            self._bind_state_artifacts(compiled, iteration)
            if iteration == 0 and self.spec.state and not self.store.initial.exists():
                state_values = {
                    name: compiled.nodes[name].value for name in self.spec.state
                }
                self.store.initialise(state_values)
                _, state_values = self.store.load()
                for name, value in state_values.items():
                    compiled.nodes[name]._value = value
            nodes = traverse_postorder(compiled.entry)
            if (
                self.spec.settings.reset_random_state
                and iteration >= self.spec.settings.reset_random_config[1]
            ):
                for node in nodes:
                    if hasattr(node, "reset_random_seed"):
                        node.reset_random_seed(mode=self.spec.settings.reset_random_config[0])

            self.state = SessionState.StepFinished
            self._run_nodes(
                iteration_directory,
                nodes_postorder=nodes,
                feed_dict={},
                reset_states=False,
                set_node_dir_func=None,
            )
            if self.state != SessionState.StepFinished:
                self._print("wait current iteration to finish...")
                return

            converged = self._converged(nodes)
            next_values = self._next_values(compiled)
            self.store.commit(iteration, next_values, converged)
            iteration_directory.mkdir(parents=True, exist_ok=True)
            (iteration_directory / "FINISHED").write_text(
                f"STATE COMMITTED FINISHED AT {time.asctime(time.localtime())}.",
                encoding="utf-8",
            )
            state_values = next_values
            last_iteration = iteration
            if converged:
                self.state = SessionState.LoopConverged
                return

        self.state = SessionState.LoopFinished
