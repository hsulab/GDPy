"""Composition preserves both arithmetic and interactive host lifecycle."""
from contextlib import contextmanager
import sys
import types

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes

from gdpx.providers import get_provider_manager
from gdpx.providers.ase.backend import EnhancedCalculator


class ConstantCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, value, events, fail=False, **kwargs):
        super().__init__(**kwargs)
        self.value, self.events, self.fail = value, events, fail

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.events.append(('calculate', self.value))
        if self.fail:
            raise RuntimeError('modifier failed')
        self.results = {'energy': self.value, 'forces': np.full((len(atoms),3), self.value), 'stress': np.full(6,self.value)}


class InteractiveCalculator(ConstantCalculator):
    @contextmanager
    def pause(self):
        self.events.append('pause')
        try:
            yield
        finally:
            self.events.append('resume')

    def finalize(self):
        self.events.append('finalize')


@pytest.mark.parametrize('interactive', [False, True])
def test_modifier_sums_properties_and_saves_host(interactive, tmp_path):
    events = []
    host = (InteractiveCalculator if interactive else ConstantCalculator)(2, events)
    modifier = ConstantCalculator(3, events)
    calc = EnhancedCalculator([host, modifier], directory=str(tmp_path))
    atoms = molecule('H2O'); atoms.calc = calc
    assert atoms.get_potential_energy() == 5
    np.testing.assert_array_equal(atoms.get_forces(), np.full((3,3),5))
    np.testing.assert_array_equal(atoms.get_stress(), np.full(6,5))
    assert calc.results['host_energy'] == 2
    np.testing.assert_array_equal(calc.results['host_forces'], np.full((3,3),2))
    assert atoms.calc is calc
    if interactive:
        assert events.index(('calculate',2)) < events.index('pause') < events.index(('calculate',3)) < events.index('resume')
    assert host.directory != modifier.directory
    calc.reset()
    assert host.results == modifier.results == {}


def test_modifier_failure_resumes_host_and_restores_atoms():
    events = []
    host = InteractiveCalculator(2, events)
    calc = EnhancedCalculator([host, ConstantCalculator(3,events,fail=True)])
    atoms = molecule('H2O'); atoms.calc = calc
    with pytest.raises(RuntimeError, match='modifier failed'):
        atoms.get_potential_energy()
    assert atoms.calc is calc
    assert events[-1] == 'resume'


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('selection', [{}, {'backend': 'ase'}, {'method': 'default'}])
def test_dftd3_modifier_runtime_and_driver_cleanup(monkeypatch, tmp_path, fail, selection):
    events = []
    import gdpx.providers.vasp.manager as vasp
    def register(self, params):
        assert params['backend'] == 'interactive'
        self.calc = InteractiveCalculator(2,events)
    monkeypatch.setattr(vasp.VaspManager, 'register_calculator', register)
    module = types.ModuleType('dftd3.ase')
    seen = []
    def dispersion(**kwargs):
        seen.append(kwargs)
        return ConstantCalculator(3,events,fail=fail)
    module.DFTD3 = dispersion
    monkeypatch.setitem(sys.modules, 'dftd3.ase', module)
    runtime = get_provider_manager().resolve_runtime({
        'potential': {'provider':'vasp','backend':'interactive'},
        'modifiers': [{'provider':'dftd3',**selection,'parameters':{'method':'PBE','damping':'d3bj'}}],
        'executor': {'provider':'ase','method':'spc'},
    })
    assert seen == [{'method':'PBE','damping':'d3bj'}]
    assert runtime.modifiers[0].backend == 'ase'
    assert runtime.modifiers[0].method == 'default'
    saved = runtime.config.to_dict()
    assert saved['modifiers'][0]['backend'] == 'ase'
    from gdpx.providers import RuntimeConfig
    assert RuntimeConfig.from_mapping(saved).to_dict() == saved
    if 'method' not in selection:
        assert 'method' not in saved['modifiers'][0]
    assert runtime.config.to_dict()['modifiers'][0]['parameters']['method'] == 'PBE'
    runtime.executor.directory = tmp_path
    atoms = molecule('H2O')
    if fail:
        with pytest.raises(RuntimeError, match='modifier failed'):
            runtime.executor.run(atoms)
    else:
        runtime.executor.run(atoms)
    assert events[-1] == 'finalize'
    assert events.count('finalize') == 1


@pytest.mark.parametrize('backend', ['bad', 'lammps'])
def test_invalid_modifier_backend_rejected_before_construction(backend, monkeypatch):
    from gdpx.providers import MaterializationError
    from gdpx.providers.dftd3.manager import Dftd3Manager
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid backend must be rejected before calculator construction')
    monkeypatch.setattr(Dftd3Manager, 'register_calculator', unexpected)
    with pytest.raises(MaterializationError, match='supported backends: ase'):
        get_provider_manager().resolve_runtime({
            'potential': {'provider': 'emt'},
            'modifiers': [{'provider': 'dftd3', 'backend': backend}],
            'executor': {'provider': 'ase', 'method': 'spc'},
        })


@pytest.mark.parametrize('backend', ['', 1, []])
def test_modifier_backend_requires_nonempty_string(backend):
    from gdpx.providers import ProviderConfigurationError, RuntimeConfig
    with pytest.raises(ProviderConfigurationError, match='Modifier backend'):
        RuntimeConfig.from_mapping({
            'potential': {'provider': 'emt'},
            'modifiers': [{'provider': 'dftd3', 'backend': backend}],
            'executor': {'provider': 'ase', 'method': 'spc'},
        })
