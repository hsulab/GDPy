"""Adapter contracts without importing or downloading an optional model."""
import copy
import sys
import subprocess
from types import ModuleType, SimpleNamespace

import pytest
from ase.calculators.calculator import Calculator

from gdpx.providers.tace.manager import TaceManager, canonicalise_tace_models


@pytest.mark.parametrize('model', [None, '', ' ', [], [''], [1]])
def test_empty_models_rejected(model):
    with pytest.raises(ValueError, match='non-empty'):
        canonicalise_tace_models(model)


def test_alias_and_local_path(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.touch()
    assert canonicalise_tace_models(['TACE-OAM-7M', str(checkpoint)], ['TACE-OAM-7M']) == [
        'TACE-OAM-7M', str(checkpoint.resolve())]
    with pytest.raises(FileNotFoundError):
        canonicalise_tace_models(str(tmp_path / 'absent.pt'))


@pytest.fixture
def fake_tace(monkeypatch, tmp_path):
    calls = []

    class FakeCalc(Calculator):
        def __init__(self, **kwargs):
            super().__init__()
            calls.append(kwargs)

    for name in ['tace', 'tace.foundations', 'tace.interface', 'tace.interface.ase']:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['tace.foundations'].tace_foundations = {'TACE-OAM-7M': tmp_path / 'cached.pt'}
    sys.modules['tace.interface.ase'].TACEAseCalc = FakeCalc
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    return calls


def test_parameters_and_serialization(fake_tace):
    params = dict(model='TACE-OAM-7M', backend='ase', device='cpu', precision='float64',
                  fidelity_idx=0, neighborlist_backend='ase')
    original = copy.deepcopy(params)
    manager = TaceManager()
    manager.register_calculator(params)
    assert params == original
    assert manager.calc_params['model'] == ['TACE-OAM-7M']
    assert fake_tace[0]['device'] == 'cpu'
    assert fake_tace[0]['dtype'] == 'float64'
    assert fake_tace[0]['fidelity_idx'] == 0
    assert fake_tace[0]['neighborlist_backend'] == 'ase'
    assert set(fake_tace[0]) == {'model', 'device', 'dtype', 'fidelity_idx', 'neighborlist_backend'}
    manager.register_calculator(manager._implementation_config()['params'])
    assert fake_tace[1] == fake_tace[0]


@pytest.mark.parametrize('uncertainty, count', [(False, 1), (True, 2)])
def test_committee(fake_tace, uncertainty, count):
    manager = TaceManager()
    manager.register_calculator(dict(model=['TACE-OAM-7M'] * 2, estimate_uncertainty=uncertainty))
    assert len(fake_tace) == count
    assert fake_tace[0]['device'] == 'cuda'
    assert fake_tace[0]['dtype'] == 'float32'


def test_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, 'tace', None)
    monkeypatch.setitem(sys.modules, 'tace.foundations', None)
    with pytest.raises(ImportError, match=r'\.\[tace\]'):
        TaceManager().register_calculator(dict(model='TACE-OAM-7M'))


def test_load_failure_is_not_masked(fake_tace):
    def broken(**kwargs):
        raise RuntimeError('invalid checkpoint')
    sys.modules['tace.interface.ase'].TACEAseCalc = broken
    with pytest.raises(RuntimeError, match='invalid checkpoint'):
        TaceManager().register_calculator(dict(model='TACE-OAM-7M'))


def test_provider_discovery_without_optional_packages():
    script = """
import sys
from importlib.abc import MetaPathFinder
class BlockOptional(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tace', 'mattersim'}:
            raise AssertionError('Optional import during discovery: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from gdpx.providers import CapabilityKind, get_provider_manager
providers = get_provider_manager()
for name in ['tace', 'mattersim']:
    providers.require(name, CapabilityKind.POTENTIAL, 'default')
    providers.require(name, CapabilityKind.MATERIALIZER, 'ase.calculator')
"""
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)
