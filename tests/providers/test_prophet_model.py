"""Prophet adapter contracts without optional packages or model downloads."""

import copy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from ase.calculators.calculator import Calculator

from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.ase.backend import CommitteeCalculator
from gdpx.providers.prophet.manager import ProphetManager, canonicalise_prophet_models


@pytest.mark.parametrize("model", [None, "", " ", [], [""], [1], 1])
def test_invalid_models(model):
    with pytest.raises(ValueError, match="non-empty checkpoint"):
        canonicalise_prophet_models(model)


def test_local_checkpoint_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    assert canonicalise_prophet_models("model.pt") == [str(checkpoint)]
    with pytest.raises(FileNotFoundError, match="does not exist"):
        canonicalise_prophet_models("absent.pt")


@pytest.fixture
def fake_prophet(monkeypatch, tmp_path):
    calls = []

    class FakeCalculator(Calculator):
        def __init__(self, **kwargs):
            super().__init__()
            calls.append(kwargs)

    module = ModuleType("prophet")
    module.KairosCalculator = FakeCalculator
    monkeypatch.setitem(sys.modules, "prophet", module)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    return checkpoint, calls


def test_parameters_and_serialization(fake_prophet):
    checkpoint, calls = fake_prophet
    params = dict(model=str(checkpoint), backend="ase", device="cpu",
                  use_kernel=False, use_compile=True, label="prophet-test")
    original = copy.deepcopy(params)
    manager = ProphetManager()
    manager.register_calculator(params)
    assert params == original
    assert manager.calc_params["model"] == [str(checkpoint.resolve())]
    assert calls == [dict(model_path=str(checkpoint.resolve()), device="cpu",
                          use_kernel=False, use_compile=True, label="prophet-test")]
    manager.register_calculator(manager._implementation_config()["params"])
    assert calls[1] == calls[0]


@pytest.mark.parametrize("cuda,expected", [(False, "cpu"), (True, "cuda")])
@pytest.mark.parametrize("device", [None, "omitted"])
def test_automatic_device_and_portable_kernel_default(fake_prophet, monkeypatch, cuda, expected, device):
    checkpoint, calls = fake_prophet
    monkeypatch.setattr(sys.modules["torch"].cuda, "is_available", lambda: cuda)
    params = dict(model=str(checkpoint))
    if device != "omitted":
        params["device"] = device
    ProphetManager().register_calculator(params)
    assert calls[0]["device"] == expected
    assert calls[0]["use_kernel"] is False


def test_explicit_cuda_kernels(fake_prophet):
    checkpoint, calls = fake_prophet
    ProphetManager().register_calculator(dict(model=str(checkpoint), device="cuda:1", use_kernel=True))
    assert calls[0]["device"] == "cuda:1"
    assert calls[0]["use_kernel"] is True


@pytest.mark.parametrize("uncertainty,count", [(False, 1), (True, 2)])
def test_committee(fake_prophet, uncertainty, count):
    checkpoint, calls = fake_prophet
    second = checkpoint.with_name("second.pt")
    second.touch()
    manager = ProphetManager()
    manager.register_calculator(dict(model=[str(checkpoint), str(second)], estimate_uncertainty=uncertainty))
    assert len(calls) == count
    assert calls[0]["model_path"] == str(checkpoint)
    assert isinstance(manager.calc, CommitteeCalculator) is uncertainty
    if uncertainty:
        assert calls[1]["model_path"] == str(second)
    assert all("estimate_uncertainty" not in call for call in calls)


def test_missing_dependency(fake_prophet, monkeypatch):
    checkpoint, _ = fake_prophet
    monkeypatch.setitem(sys.modules, "prophet", None)
    with pytest.raises(ImportError, match=r"\.\[prophet\]"):
        ProphetManager().register_calculator(dict(model=str(checkpoint)))


def test_checkpoint_failure_is_not_masked(fake_prophet, monkeypatch):
    checkpoint, _ = fake_prophet

    def broken(**kwargs):
        raise RuntimeError("invalid checkpoint")

    monkeypatch.setattr(sys.modules["prophet"], "KairosCalculator", broken)
    with pytest.raises(RuntimeError, match="invalid checkpoint"):
        ProphetManager().register_calculator(dict(model=str(checkpoint)))


def test_documented_runtime(fake_prophet):
    from pathlib import Path

    checkpoint, calls = fake_prophet
    guide = Path(__file__).parents[2] / "docs/source/potentials/prophet.md"
    example = guide.read_text().split("```yaml\n", 1)[1].split("```", 1)[0]
    config = yaml.safe_load(example)
    config["potential"]["parameters"]["model"] = str(checkpoint)
    runtime = get_provider_manager().resolve_runtime(config)
    assert runtime.potential.backend == "ase"
    assert isinstance(runtime.materialization.calculator, Calculator)
    assert calls[0]["model_path"] == str(checkpoint)
    assert calls[0]["device"] == "cpu"
    assert get_provider_manager().supports("prophet", CapabilityKind.MATERIALIZER, "ase.calculator")
    assert not get_provider_manager().supports("prophet", CapabilityKind.TRAINER)


def test_discovery_without_optional_packages():
    script = """
import sys
from importlib.abc import MetaPathFinder
class BlockOptional(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'prophet', 'prophet_spin'}:
            raise AssertionError('Optional import during discovery: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from gdpx.providers import CapabilityKind, get_provider_manager
manager = get_provider_manager()
manager.require('prophet', CapabilityKind.POTENTIAL, 'default')
manager.require('prophet', CapabilityKind.MATERIALIZER, 'ase.calculator')
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
