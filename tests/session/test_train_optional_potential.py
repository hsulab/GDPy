from types import SimpleNamespace

import pytest
import yaml

from gdpx.providers import ComponentConfig
from gdpx.workflow.compiler import compile_workflow, validate_workflow
from gdpx.workflow.configuration import load_workflow
from gdpx.workflow.nodes.runtime import PotentialVariable
from gdpx.workflow.nodes.trainer import train
from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.workflow.session.variable import Variable


@registers.variable.register
class TrainingTestVariable(Variable):
    def __init__(self, value=None, directory="."):
        super().__init__(value, directory)


def _trainer():
    variable = Variable(SimpleNamespace(type_list=["Co", "H", "O", "Ti"]))
    variable.config = ComponentConfig("deepmd")
    return variable


@pytest.mark.parametrize("size", [1, 4])
def test_omitted_potential_starts_from_scratch_each_time(monkeypatch, tmp_path, size):
    calls = []
    models = [f"m{i}/deepmd-c.pt2" for i in range(size)]

    class Worker:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, dataset, **kwargs):
            calls.append(kwargs["init_models"])

        def inspect(self, **kwargs):
            pass

        def get_number_of_running_jobs(self):
            return 0

        def retrieve(self, **kwargs):
            return models

    monkeypatch.setattr("gdpx.workflow.nodes.trainer.TrainerBasedWorker", Worker)
    trainer = _trainer()
    operation = train(Variable("dataset"), trainer, size=size, directory=tmp_path)
    potential = operation.input_nodes[3].value

    # Reusing the operation must not initialize from its previous output models.
    for _ in range(2):
        result = operation.forward("dataset", trainer.value, None, potential)
        assert result.provider == "deepmd"
        assert list(result.parameters["type_list"]) == trainer.value.type_list
        assert list(result.parameters["model"]) == models
        assert result.parameters["estimate_uncertainty"] is (size > 1)
    assert calls == [[None] * size, [None] * size]
    assert "model" not in potential.parameters


def test_explicit_potential_keeps_model_initialization_and_backend(monkeypatch, tmp_path):
    models = ["m0/model.ckpt.pt", "m1/model.ckpt.pt"]
    calls = []

    class Worker:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, dataset, **kwargs):
            calls.append(kwargs["init_models"])

        def inspect(self, **kwargs):
            pass

        def get_number_of_running_jobs(self):
            return 0

        def retrieve(self, **kwargs):
            return ["new0.pt2", "new1.pt2"]

    monkeypatch.setattr("gdpx.workflow.nodes.trainer.TrainerBasedWorker", Worker)
    trainer = _trainer()
    potential = PotentialVariable("deepmd", backend="lammps", parameters={
        "model": models, "type_list": trainer.value.type_list,
        "estimate_uncertainty": True,
    })
    operation = train(Variable("dataset"), trainer, potential, size=2, directory=tmp_path)
    result = operation.forward("dataset", trainer.value, None, potential.value)

    assert calls == [models]
    assert result.backend == "lammps"
    assert result.parameters["estimate_uncertainty"] is True
    assert list(potential.value.parameters["model"]) == models


@pytest.mark.parametrize("parameters,provider,message", [
    ({}, "other", "providers must match"),
    ({"type_list": ["H"]}, "deepmd", "type lists must match"),
])
def test_explicit_potential_still_checks_compatibility(parameters, provider, message):
    with pytest.raises(ValueError, match=message):
        train(Variable("dataset"), _trainer(), PotentialVariable(provider, parameters=parameters))


def test_yaml_without_potential_validates_and_compiles(monkeypatch, tmp_path):
    monkeypatch.setattr("gdpx.workflow.nodes.trainer.create_trainer", lambda config: _trainer().value)
    payload = {
        "workflow": {"targets": "train"},
        "resources": {
            "dataset": {"__type__": "training_test", "options": {}},
            "trainer": {"__type__": "trainer", "options": {"provider": "deepmd"}},
        },
        "steps": {"train": {
            "__type__": "train", "inputs": {"dataset": "dataset", "trainer": "trainer"},
            "options": {"size": 4},
        }},
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(payload))
    spec = load_workflow(path)
    validate_workflow(spec)
    compiled = compile_workflow(spec, tmp_path / "run")
    potential = compiled.nodes["train"].input_nodes[3].value
    assert potential.provider == "deepmd"
    assert "model" not in potential.parameters
    assert potential.parameters["estimate_uncertainty"] is True
