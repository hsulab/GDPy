"""Backend routing is declared by providers, independent of optional engines."""
import importlib
from types import SimpleNamespace

import pytest
from ase.build import molecule
from ase.calculators.emt import EMT

from gdpx.providers import CapabilityKind, MaterializationError, RuntimeConfig, get_provider_manager
from gdpx.providers.adapters import BackendMaterializer, select_backend


def routes():
    providers = get_provider_manager()
    return [(name, target, mat) for name in providers
            for target, mat in providers.get(name).implementations(CapabilityKind.MATERIALIZER).items()]


@pytest.mark.parametrize('name,target,mat', routes(), ids=lambda x: x if isinstance(x, str) else type(x).__name__)
def test_every_builtin_declares_and_validates_default(name, target, mat):
    default = select_backend(mat)
    assert isinstance(default, str) and default
    assert select_backend(mat, default) == default
    with pytest.raises(MaterializationError, match='supported backends'):
        select_backend(mat, 'nonexistent')


@pytest.mark.parametrize('name,target,default,alternatives', [
    ('reax', 'ase.calculator', 'xreac', ('xreac', 'reax/c')),
    ('reax', 'lammps.potential', 'reax/c', ('reax/c',)),
    ('deepmd', 'ase.calculator', 'ase', ('ase', 'lammps')),
    ('nequip', 'ase.calculator', 'ase', ('ase', 'lammps')),
    ('allegro', 'ase.calculator', 'ase', ('ase', 'lammps')),
    ('beann', 'ase.calculator', 'ase', ('ase', 'lammps')),
    ('mattersim', 'ase.calculator', 'ase', ('ase', 'graph_pes')),
    ('xtb', 'ase.calculator', 'xtb', ('xtb', 'tblite')),
    ('cp2k', 'ase.calculator', 'cp2k', ('cp2k', 'interactive')),
    ('vasp', 'ase.calculator', 'interactive', ('interactive',)),
    ('deepmd_jax', 'ase.calculator', 'ase', ('ase', 'jax')),
])
def test_supported_alternatives_dispatch(name, target, default, alternatives, monkeypatch):
    mat = get_provider_manager().require(name, CapabilityKind.MATERIALIZER, target)
    assert select_backend(mat) == default
    assert tuple(mat.backends) == alternatives
    calls = []
    for backend in alternatives:
        delegate = mat.backends[backend]
        monkeypatch.setattr(type(delegate), 'materialize', lambda self, potential, target, **kw: calls.append(self) or 'result')
        assert mat.materialize(object(), target, backend=backend) == 'result'
        assert calls[-1] is delegate


@pytest.mark.parametrize('backend', [None, 'ase'])
def test_resolved_backend_persists(backend):
    potential = {'provider': 'emt'}
    if backend is not None:
        potential['backend'] = backend
    runtime = get_provider_manager().resolve_runtime({'potential': potential, 'executor': {'provider': 'ase', 'method': 'spc'}})
    assert runtime.potential.backend == 'ase'
    saved = runtime.config.to_dict()
    assert saved['potential']['backend'] == 'ase'
    assert RuntimeConfig.from_mapping(saved).to_dict() == saved
    assert get_provider_manager().resolve_runtime(saved).potential == runtime.potential


@pytest.mark.parametrize('potential,match', [
    ({'provider': 'vasp', 'backend': 'vasp_interactive'}, 'renamed'),
    ({'provider': 'cp2k', 'backend': 'cp2k_shell'}, 'renamed'),
    ({'provider': 'vasp', 'backend': 'vasp_interactive_disp'}, 'dftd3 modifier'),
    ({'provider': 'vasp', 'parameters': {'dispersion': {}}}, 'dftd3 modifier'),
    ({'provider': 'cp2k', 'parameters': {'interface': 'cp2k_shell'}}, 'potential.backend'),
    ({'provider': 'emt', 'parameters': {'backend': 'ase'}}, 'potential.backend'),
])
def test_removed_configuration_has_migration(potential, match):
    with pytest.raises(ValueError, match=match):
        RuntimeConfig.from_mapping({'potential': potential, 'executor': {'provider': 'ase', 'method': 'spc'}})


@pytest.mark.parametrize('name,backend', [('vasp','interactive'), ('cp2k','interactive'), ('cp2k','cp2k')])
def test_interactive_materializer_dispatch(name, backend, monkeypatch):
    module = importlib.import_module(f'gdpx.providers.{name}.manager')
    manager_cls = getattr(module, {'vasp':'VaspManager', 'cp2k':'Cp2kManager'}[name])
    calls = []
    def register(self, params):
        calls.append(params)
        self.calc = EMT()
    monkeypatch.setattr(manager_cls, 'register_calculator', register)
    runtime = get_provider_manager().resolve_runtime({
        'potential': {'provider': name, 'backend': backend},
        'executor': {'provider': 'ase', 'method': 'spc'},
    })
    assert calls == [{'backend': backend}]
    assert runtime.potential.backend == backend


def test_lammps_reax_with_ase_is_single_point(tmp_path):
    model = tmp_path/'ffield'
    model.touch()
    runtime = get_provider_manager().resolve_runtime({
        'potential': {'provider': 'reax', 'backend': 'reax/c', 'parameters': {
            'model': str(model), 'command': 'lmp', 'type_list': ['H','O'],
            'task': 'md', 'steps': 999, 'dynamics': ['fix motion all nve']}},
        'executor': {'provider': 'ase', 'method': 'md', 'parameters': {'steps': 2}},
    })
    calc = runtime.materialization.calculator
    assert calc.task == 'spc'
    assert calc._write_simulation_tasks().strip() == 'run             0'
    atoms = molecule('H2O'); atoms.center(vacuum=5); atoms.pbc = True
    calc.directory = str(tmp_path/'calc')
    calc.type_list = ["H", "O"]
    calc.write_input(atoms)
    text = (tmp_path/'calc'/'in.lammps').read_text()
    assert 'qeq/reax' in text and 'pair_style' in text
    assert 'minimize' not in text and 'fix motion' not in text
    assert runtime.potential.backend == 'reax/c'


def test_invalid_backend_fails_without_loading_engine():
    with pytest.raises(MaterializationError, match='supported backends: xreac, reax/c'):
        get_provider_manager().resolve_runtime({
            'potential': {'provider': 'reax', 'backend': 'bad'},
            'executor': {'provider': 'ase', 'method': 'spc'},
        })


def test_legacy_third_party_materializer_requires_declaration_for_override():
    mat = SimpleNamespace(materialize=lambda *args: object())
    assert select_backend(mat) is None
    with pytest.raises(MaterializationError, match='not declared'):
        select_backend(mat, 'ase')
