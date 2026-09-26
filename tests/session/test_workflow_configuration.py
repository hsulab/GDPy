import json
import pathlib

import pytest
import yaml

from gdpx import config
from gdpx.cli.workflow import parse_overrides, show_workflow_status
from gdpx.workflow.compiler import compile_workflow, workflow_dot, workflow_plan
from gdpx.workflow.configuration import WorkflowConfigError, load_workflow
from gdpx.workflow.session.interface import run_workflow_spec
from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.workflow.session.variable import Variable
from gdpx.workflow.state import NamedOutputs


@registers.variable.register
class WorkflowTestVariable(Variable):
    def __init__(self, value, directory="."):
        super().__init__(value, directory)


@registers.operation.register
class workflow_test_add(Operation):
    def __init__(self, value, amount=1, directory="."):
        super().__init__([value], directory)
        self.amount = amount

    def forward(self, value):
        super().forward()
        self.status = "finished"
        return value + self.amount


@registers.operation.register
class workflow_test_join(Operation):
    def __init__(self, nodes, directory="."):
        super().__init__(nodes, directory)


@registers.operation.register
class workflow_test_pair(Operation):
    output_names = ("left", "right")

    def __init__(self, value, directory="."):
        super().__init__([value], directory)

    def forward(self, value):
        super().forward()
        self.status = "finished"
        return NamedOutputs(left=value + 1, right=value + 2)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_load_compile_plan_and_graph(tmp_path):
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"amount": 4}), encoding="utf-8")
    _write(
        tmp_path / "resources.yaml",
        """
parameters:
  start: 2
  payload: {$file: payload.json}
resources:
  value:
    __type__: workflow_test
    options:
      value: {$param: start}
""",
    )
    workflow_path = _write(
        tmp_path / "workflow.yaml",
        """
includes: [resources.yaml]
profiles:
  larger:
    parameters:
      start: 3
workflow:
  targets: result
resources:
  amount:
    __type__: workflow_test
    options:
      value: {$param: payload.amount}
steps:
  result:
    __type__: workflow_test_add
    inputs:
      value: value
    options:
      amount: {$param: payload.amount}
""",
    )

    spec = load_workflow(workflow_path, profile="larger")
    compiled = compile_workflow(spec, tmp_path / "run")

    assert compiled.nodes["value"].value == 3
    assert compiled.nodes["amount"].value == 4
    assert compiled.entry.input_nodes == [compiled.nodes["value"]]
    assert "result [step:workflow_test_add] <- value" in workflow_plan(spec)
    assert '"value" -> "result"' in workflow_dot(spec)


def test_parameter_override_is_typed(tmp_path):
    path = _write(
        tmp_path / "workflow.yaml",
        """
parameters: {start: 2}
workflow: {targets: result}
resources:
  value:
    __type__: workflow_test
    options: {value: {$param: start}}
steps:
  result:
    __type__: workflow_test_add
    inputs: {value: value}
""",
    )

    spec = load_workflow(path, overrides={"start": 7})

    assert spec.resources["value"].options["value"] == 7


def test_workflow_executor_resource_broadcasts_runtime_parameters(tmp_path):
    path = _write(
        tmp_path / "workflow.yaml",
        """
workflow: {targets: result}
resources:
  potential:
    __type__: potential
    options: {provider: emt}
  executor:
    __type__: executor
    options:
      provider: ase
      method: md
      parameters: {ensemble: nvt, steps: 1}
      broadcast: {temp: [300, 600]}
  runtime:
    __type__: runtime
    inputs: {potential: potential, executor: executor}
steps:
  result:
    __type__: workflow_test_add
    inputs: {value: runtime}
    options: {amount: 0}
""",
    )

    compiled = compile_workflow(load_workflow(path), tmp_path / "run")

    runtimes = compiled.nodes["runtime"].value
    assert [runtime.config.executor.parameters["temp"] for runtime in runtimes] == [300, 600]


def test_compiled_workflow_runs_with_stable_directory(tmp_path):
    path = _write(
        tmp_path / "example.yaml",
        """
workflow: {targets: result}
resources:
  value: {__type__: workflow_test, options: {value: 2}}
steps:
  result: {__type__: workflow_test_add, inputs: {value: value}, options: {amount: 3}}
""",
    )

    spec = load_workflow(path)
    assert spec.settings.mode == "once"
    assert run_workflow_spec(spec, tmp_path / "runs")
    assert (tmp_path / "runs" / "example" / "FINISHED").is_file()


def test_repeat_workflow_runs_each_iteration(tmp_path):
    path = _write(
        tmp_path / "repeated.yaml",
        """
workflow: {mode: repeat, targets: result, max_iterations: 2}
resources:
  value: {__type__: workflow_test, options: {value: 2}}
steps:
  result: {__type__: workflow_test_add, inputs: {value: value}}
""",
    )

    spec = load_workflow(path)
    assert spec.settings.mode == "repeat"
    assert run_workflow_spec(spec, tmp_path / "runs")
    assert (tmp_path / "runs" / "repeated" / "iter.0000" / "FINISHED").is_file()
    assert (tmp_path / "runs" / "repeated" / "iter.0001" / "FINISHED").is_file()


def test_repeat_workflow_commits_named_state_outputs(tmp_path, monkeypatch):
    path = _write(
        tmp_path / "stateful.yaml",
        """
workflow: {mode: repeat, targets: result, max_iterations: 3}
state:
  counter:
    initial: seed
    update: {node: pair, output: right}
resources:
  seed: {__type__: workflow_test, options: {value: 0}}
steps:
  pair: {__type__: workflow_test_pair, inputs: {value: counter}}
  result:
    __type__: workflow_test_add
    inputs: {value: {node: pair, output: left}}
    options: {amount: 0}
""",
    )

    assert run_workflow_spec(load_workflow(path), tmp_path / "runs")
    run = tmp_path / "runs" / "stateful"
    manifest = yaml.safe_load((run / "state" / "current.yaml").read_text())

    assert manifest["iteration"] == 2
    assert manifest["values"]["counter"] == {"kind": "json", "value": 6}
    assert (run / "iter.0002" / "steps" / "pair").is_dir()

    messages = []
    monkeypatch.setattr(config, "_print", messages.append)
    show_workflow_status(path, directory=tmp_path / "runs")
    assert "iteration: 2" in messages[-1]
    assert "counter: int" in messages[-1]


def test_repeat_workflow_rejects_changed_configuration_on_resume(tmp_path):
    path = _write(
        tmp_path / "stateful.yaml",
        """
workflow: {mode: repeat, targets: result, max_iterations: 1}
steps:
  result: {__type__: workflow_test_add, inputs: {value: value}}
resources:
  value: {__type__: workflow_test, options: {value: 1}}
""",
    )
    assert run_workflow_spec(load_workflow(path), tmp_path / "runs")
    path.write_text(path.read_text().replace("value: 1", "value: 2"))

    with pytest.raises(RuntimeError, match="configuration changed"):
        run_workflow_spec(load_workflow(path), tmp_path / "runs")


def test_cli_overrides_use_yaml_types():
    assert parse_overrides(["count=3", "enabled=true", "names=[a, b]"]) == {
        "count": 3,
        "enabled": True,
        "names": ["a", "b"],
    }

    with pytest.raises(WorkflowConfigError, match="KEY=VALUE"):
        parse_overrides(["count"])


def test_nested_input_lists_are_rendered_consistently(tmp_path):
    path = _write(
        tmp_path / "workflow.yaml",
        """
workflow: {targets: result}
resources:
  first: {__type__: workflow_test, options: {value: 1}}
  second: {__type__: workflow_test, options: {value: 2}}
steps:
  result:
    __type__: workflow_test_join
    inputs:
      nodes: [[first], second]
""",
    )

    spec = load_workflow(path)

    assert "result [step:workflow_test_join] <- first, second" in workflow_plan(spec)
    assert '"first" -> "result"' in workflow_dot(spec)
    assert '"second" -> "result"' in workflow_dot(spec)


def test_active_learning_example_uses_validation_targets():
    root = pathlib.Path(__file__).resolve().parents[2]
    spec = load_workflow(root / "examples" / "workflows" / "active-learning.yaml")

    assert spec.settings.mode == "repeat"
    assert spec.settings.targets == ("test_spc_train", "test_spc_test")
    assert set(spec.state) == {"current_potential", "training_data", "test_data"}
    assert "shared/" not in spec.source.read_text()
    assert list(spec.steps) == [
        "read_stru",
        "model_to_explore",
        "run_nvt",
        "est_devi",
        "select_devi",
        "select_desc",
        "run_vasp",
        "sift_forces",
        "transfer",
        "train",
        "model_spc",
        "test_spc_train",
        "test_spc_test",
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("schema_version: 1", "remove `schema_version`"),
        (
            """
workflow: {mode: active, targets: result}
steps:
  result: {__type__: workflow_test_add, inputs: {value: result}}
""",
            "`once` or `repeat`",
        ),
        (
            """
workflow: {mode: once, targets: result, max_iterations: 2}
steps:
  result: {__type__: workflow_test_add, inputs: {value: result}}
""",
            "only in `repeat` mode",
        ),
        (
            """
workflow: {targets: missing}
steps:
  result: {__type__: workflow_test_add, inputs: {value: missing}}
""",
            "references unknown nodes",
        ),
        (
            """
workflow: {targets: first}
steps:
  first: {__type__: workflow_test_add, inputs: {value: second}}
  second: {__type__: workflow_test_add, inputs: {value: first}}
""",
            "dependency cycle",
        ),
        (
            """
workflow: {targets: result}
resources:
  data: {__type__: workflow_test, inputs: {value: result}}
steps:
  result: {__type__: workflow_test_add, inputs: {value: data}}
""",
            "cannot depend on steps",
        ),
        (
            """
workflow: {targets: result}
resources:
  value: {__type__: workflow_test, options: {value: 1}}
steps:
  result:
    __type__: workflow_test_add
    inputs: {value: value}
    options: {value: 2}
""",
            "both inputs and options",
        ),
        (
            """
workflow: {targets: [result, result]}
resources:
  value: {__type__: workflow_test, options: {value: 1}}
steps:
  result: {__type__: workflow_test_add, inputs: {value: value}}
""",
            "cannot contain duplicates",
        ),
        (
            """
workflow: {targets: result}
state:
  value: {initial: seed, update: result}
resources:
  seed: {__type__: workflow_test, options: {value: 1}}
steps:
  result: {__type__: workflow_test_add, inputs: {value: value}}
""",
            "valid only",
        ),
        (
            """
workflow: {mode: repeat, targets: result}
state:
  value: {initial: seed, update: unused}
resources:
  seed: {__type__: workflow_test, options: {value: 1}}
steps:
  result: {__type__: workflow_test_add, inputs: {value: value}}
  unused: {__type__: workflow_test_add, inputs: {value: value}}
""",
            "not reachable",
        ),
    ],
)
def test_invalid_workflows_have_contextual_errors(tmp_path, body, message):
    path = _write(tmp_path / "workflow.yaml", body)

    with pytest.raises(WorkflowConfigError, match=message):
        load_workflow(path)


def test_include_cycles_and_duplicate_nodes_are_rejected(tmp_path):
    _write(tmp_path / "a.yaml", "includes: [b.yaml]\n")
    _write(tmp_path / "b.yaml", "includes: [a.yaml]\n")
    with pytest.raises(WorkflowConfigError, match="include cycle"):
        load_workflow(tmp_path / "a.yaml")

    _write(tmp_path / "fragment.yaml", "resources: {value: {__type__: workflow_test, options: {value: 1}}}\n")
    _write(
        tmp_path / "workflow.yaml",
        """
includes: [fragment.yaml]
workflow: {targets: result}
resources: {value: {__type__: workflow_test, options: {value: 2}}}
steps: {result: {__type__: workflow_test_add, inputs: {value: value}}}
""",
    )
    with pytest.raises(WorkflowConfigError, match="duplicate resources: value"):
        load_workflow(tmp_path / "workflow.yaml")


def test_unknown_named_output_is_rejected_before_construction(tmp_path):
    path = _write(
        tmp_path / "workflow.yaml",
        """
workflow: {targets: result}
resources:
  value: {__type__: workflow_test, options: {value: 1}}
steps:
  pair: {__type__: workflow_test_pair, inputs: {value: value}}
  result:
    __type__: workflow_test_add
    inputs: {value: {node: pair, output: missing}}
""",
    )

    with pytest.raises(WorkflowConfigError, match="has no output"):
        compile_workflow(load_workflow(path), tmp_path / "run")
