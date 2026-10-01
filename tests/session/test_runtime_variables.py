import pytest
from ase import Atoms

from gdpx.data.array import AtomsNDArray
from gdpx.execution import Runtime
from gdpx.providers import ComponentConfig, PotentialConfig
from gdpx.workflow.nodes.runtime import (
    ExecutorVariable,
    PotentialVariable,
    RuntimeChainVariable,
    RuntimeVariable,
)
from gdpx.workflow.nodes.data import assemble
from gdpx.workflow.nodes.validator import validate
from gdpx.workflow.session.variable import Variable


def test_runtime_variable_resolves_declarative_components():
    potential = PotentialVariable("emt", parameters={})
    executor = ExecutorVariable("ase", "min", {"steps": 1})
    variable = RuntimeVariable(potential, executor)

    assert isinstance(potential.value, ComponentConfig)
    assert isinstance(executor.value, ComponentConfig)
    assert isinstance(variable.value, Runtime)
    assert variable.as_dict()["schema_version"] == 4


def test_potential_variable_preserves_backend_selection():
    potential = PotentialVariable("deepmd", backend="lammps", parameters={"model": ["model.pb"]})

    assert isinstance(potential.value, PotentialConfig)
    assert potential.value.backend == "lammps"
    assert potential.as_dict()["backend"] == "lammps"


def test_validate_operation_creates_worker_from_runtime(tmp_path):
    class RecordingValidator:
        def run(self, dataset, worker, **kwargs):
            self.dataset = dataset
            self.worker = worker
            return True

    frames = AtomsNDArray([Atoms("H")])
    validator = RecordingValidator()
    runtime = RuntimeVariable(PotentialVariable("emt"), ExecutorVariable("ase", "spc"))
    operation = validate(Variable(frames), Variable(validator), runtime, directory=tmp_path)

    operation.forward(frames, validator, runtime.value)

    assert operation.status == "finished"
    assert validator.worker.runtime is runtime.value
    assert list(validator.dataset) == ["reference"]


def test_runtime_chain_is_explicit_and_ordered():
    runtime = RuntimeVariable(
        PotentialVariable("emt"),
        ExecutorVariable("ase", "min", {"steps": 1}),
    )
    chain = RuntimeChainVariable([runtime, runtime])

    assert chain.value == (runtime.value, runtime.value)
    assert len(chain.as_dict()) == 2


def test_assemble_runtime_chain_preserves_nested_runtime_variable(tmp_path):
    runtime = _swept_runtime([300, 600])
    operation = assemble(
        variable="runtime_chain",
        runtimes=[runtime],
        directory=tmp_path,
    )

    result = operation.forward()

    assert len(result) == 2
    assert [chain[0].config.executor.parameters["temp"] for chain in result] == [300, 600]


def _swept_runtime(values):
    return RuntimeVariable(
        PotentialVariable("emt"),
        ExecutorVariable(
            "ase",
            "md",
            {"ensemble": "nvt", "steps": 1},
            broadcast={"temp": values},
        ),
    )


def test_runtime_variable_expands_executor_broadcast():
    executor = ExecutorVariable(
        "ase",
        "md",
        {"ensemble": "nvt", "steps": 1},
        broadcast={"temp": [300, 600]},
    )

    variable = RuntimeVariable(PotentialVariable("emt"), executor)

    assert isinstance(variable.value, tuple)
    assert [runtime.config.executor.parameters["temp"] for runtime in variable.value] == [300, 600]
    assert executor.as_dict()["broadcast"] == {"temp": [300, 600]}
    assert all("broadcast" not in item["executor"] for item in variable.as_dict())


def test_runtime_variable_expands_modifier_broadcast():
    variable = RuntimeVariable.from_mapping(
        {
            "potential": {"provider": "emt"},
            "modifiers": [
                {
                    "provider": "builtin",
                    "method": "distance_harmonic",
                    "parameters": {"group": [0, 1], "kspring": 5.0},
                    "broadcast": {"center": [1.0, 1.5]},
                }
            ],
            "executor": {
                "provider": "ase",
                "method": "min",
                "parameters": {"steps": 1},
            },
        }
    )

    assert isinstance(variable.value, tuple)
    assert [runtime.config.modifiers[0].parameters["center"] for runtime in variable.value] == [1.0, 1.5]
    assert all("broadcast" not in item["modifiers"][0] for item in variable.as_dict())


def test_runtime_chain_pairs_broadcasts_and_repeats_scalar_steps():
    scalar = RuntimeVariable(
        PotentialVariable("emt"),
        ExecutorVariable("ase", "min", {"steps": 1}),
    )
    chain = RuntimeChainVariable([_swept_runtime([300, 600]), scalar, _swept_runtime([400, 700])])

    assert len(chain.value) == 2
    assert [step.config.executor.parameters.get("temp") for step in chain.value[0]] == [300, None, 400]
    assert [step.config.executor.parameters.get("temp") for step in chain.value[1]] == [600, None, 700]


def test_runtime_chain_rejects_incompatible_broadcast_widths():
    with pytest.raises(ValueError, match="step 0: 2"):
        RuntimeChainVariable([_swept_runtime([300, 600]), _swept_runtime([300, 600, 900])])
