import pytest

from gdpx.execution import Runtime
from gdpx.providers import ComponentConfig
from gdpx.workflow.nodes.runtime import (
    ExecutorVariable,
    PotentialVariable,
    RuntimeChainVariable,
    RuntimeVariable,
)


def test_runtime_variable_resolves_declarative_components():
    potential = PotentialVariable("emt", parameters={})
    executor = ExecutorVariable("ase", "min", {"steps": 1})
    variable = RuntimeVariable(potential, executor)

    assert isinstance(potential.value, ComponentConfig)
    assert isinstance(executor.value, ComponentConfig)
    assert isinstance(variable.value, Runtime)
    assert variable.as_dict()["schema_version"] == 4


def test_runtime_chain_is_explicit_and_ordered():
    runtime = RuntimeVariable(
        PotentialVariable("emt"),
        ExecutorVariable("ase", "min", {"steps": 1}),
    )
    chain = RuntimeChainVariable([runtime, runtime])

    assert chain.value == (runtime.value, runtime.value)
    assert len(chain.as_dict()) == 2


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
