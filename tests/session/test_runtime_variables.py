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
    assert variable.as_dict()["schema_version"] == 2


def test_runtime_chain_is_explicit_and_ordered():
    runtime = RuntimeVariable(
        PotentialVariable("emt"),
        ExecutorVariable("ase", "min", {"steps": 1}),
    )
    chain = RuntimeChainVariable([runtime, runtime])

    assert chain.value == (runtime.value, runtime.value)
    assert len(chain.as_dict()) == 2
