from pathlib import Path
from types import SimpleNamespace

import pytest
from ase.build import bulk

from gdpx.execution import resolve_runtime
from gdpx.providers import RuntimeConfig
from gdpx.workflow.nodes.driver import compute
from gdpx.workflow.session.variable import Variable


@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
def test_compute_uncertainty_override_isolated_and_materialized(
    tmp_path, monkeypatch, override, initial, broadcast
):
    models = [tmp_path / f"m{i}.pt2" for i in range(4)]
    for model in models:
        model.write_bytes(b"fixture")
    config = RuntimeConfig.from_mapping({
        "potential": {
            "provider": "deepmd",
            "parameters": {
                "model": [str(model) for model in models],
                "type_list": ["Cu"],
                "estimate_uncertainty": initial,
            },
        },
        "executor": {"provider": "lammps", "method": "md",
                     "parameters": {"ensemble": "nvt", "steps": 1}},
        "dispatch": {"batch_size": 4},
    })
    runtime = resolve_runtime(config)
    before = runtime.config.to_dict()
    workers = []

    class PendingWorker:
        def __init__(self, value):
            self.runtime = value
            self.driver = SimpleNamespace(setting=SimpleNamespace(task="md"))
            workers.append(self)

        def run(self, frames):
            pass

        def inspect(self, resubmit=False):
            pass

        def get_number_of_running_jobs(self):
            return 1

    monkeypatch.setattr("gdpx.workflow.nodes.driver.create_worker", PendingWorker)
    monkeypatch.setattr("gdpx.workflow.nodes.driver.create_workers",
                        lambda values: [PendingWorker(value) for value in values])
    values = (runtime, runtime) if broadcast else runtime
    operation = compute(Variable(values), structures=Variable([]),
                        estimate_uncertainty=override, directory=tmp_path / "step")
    operation.forward([bulk("Cu")], values)

    expected = initial if override is None else override
    assert len(workers) == (2 if broadcast else 1)
    for worker in workers:
        resolved = worker.runtime
        assert resolved.config.potential.parameters["estimate_uncertainty"] is expected
        assert resolved.config.dispatch == runtime.config.dispatch
        # Check the actual LAMMPS input, rather than only the override flag.
        calc = resolved.materialization.calculator
        calc.type_list = ["Cu"]
        calc.directory = str(tmp_path / f"input-{len(workers)}")
        calc.write_input(bulk("Cu"))
        pair_style = next(line for line in
                          (Path(calc.directory) / "in.lammps").read_text().splitlines()
                          if line.startswith("pair_style"))
        assert str(models[0]) in pair_style
        for model in models[1:]:
            assert (str(model) in pair_style) is expected
        # Serialization preserves the override for dispatched jobs.
        saved = RuntimeConfig.from_mapping(resolved.config.to_dict())
        assert saved.potential.parameters["estimate_uncertainty"] is expected
        if override is None:
            assert resolved is runtime
    assert runtime.config.to_dict() == before
    assert config.potential.parameters["estimate_uncertainty"] is initial


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_compute_rejects_nonboolean_uncertainty(value):
    with pytest.raises(TypeError, match="boolean or None"):
        compute(Variable(None), structures=Variable([]), estimate_uncertainty=value)
