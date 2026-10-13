from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from gdpx.providers.mattersim.manager import MatterSimManager, canonicalise_mattersim_model


@pytest.mark.parametrize("backend", ["ase", "graph_pes"])
@pytest.mark.parametrize("cuda_available", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:1", None, "omitted"])
def test_device_selection(backend, cuda_available, device, monkeypatch):
    calls = []

    def calculator(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules, "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available)),
    )
    for name in ("mattersim", "mattersim.forcefield", "graph_pes",
                 "graph_pes.interfaces", "graph_pes.utils", "graph_pes.utils.calculator"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["mattersim.forcefield"].MatterSimCalculator = SimpleNamespace(from_checkpoint=calculator)
    sys.modules["graph_pes.interfaces"].mattersim = lambda **kwargs: object()
    sys.modules["graph_pes.utils.calculator"].GraphPESCalculator = lambda model, **kwargs: calculator(**kwargs)

    params = {"backend": backend, "model": "MatterSim-v1.0.0-1M"}
    if device != "omitted":
        params["device"] = device
    manager = MatterSimManager()
    manager.register_calculator(params)
    expected = device if device not in (None, "omitted") else "cuda" if cuda_available else "cpu"
    assert calls[0]["device"] == expected
    if device != "omitted":
        assert manager.calc_params["device"] == device


@pytest.mark.parametrize(
    "model",
    [
        "MatterSim-v1.0.0-1M",
        "MatterSim-v1.0.0-1M.pth",
        "MatterSim-v1.0.0-5M",
        "MatterSim-v1.0.0-5M.pth",
    ],
)
def test_pretrained_model_names_are_preserved(model):
    assert canonicalise_mattersim_model(model) == model


def test_local_model_path_is_resolved(tmp_path):
    model = tmp_path / "model.pth"
    model.touch()

    assert canonicalise_mattersim_model(str(model)) == str(model.resolve())


@pytest.mark.parametrize("model", ["", None])
def test_model_must_be_a_nonempty_string(model):
    with pytest.raises(ValueError, match="non-empty"):
        canonicalise_mattersim_model(model)


def test_missing_local_model_is_rejected(tmp_path):
    model = Path(tmp_path, "missing.pth")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        canonicalise_mattersim_model(str(model))


def test_water_cluster_example_and_small_mattersim_runtime():
    path = (
        Path(__file__).parents[2]
        / "examples"
        / "global_optimisation"
        / "explorations/genetic_algorithm/water4.yaml"
    )
    with path.open() as stream:
        config = yaml.safe_load(stream)

    builder = config["system"]["builders"]["random"]
    assert builder["composition"] == {"H2O": 4}
    assert config["system"].get("periodic", True) is True
    assert config["system"].get("preserve_fragments", True) is True
    assert config["strategy"]["operators"]["crossover"]["method"] == (
        "cut_and_splice"
    )
    assert config["strategy"]["operators"]["mutation"]["method"] == "rattle"
    runtime_path = Path(__file__).parents[2] / "examples/global_optimisation/runtimes/mattersim.yaml"
    runtime = yaml.safe_load(runtime_path.read_text())
    assert runtime["potential"] == {
        "provider": "mattersim",
        "parameters": {
            "model": "MatterSim-v1.0.0-1M",
            "compute_stress": False,
        },
    }
