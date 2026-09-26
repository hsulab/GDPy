"""NequIP and Allegro are independent potential families."""
import copy

import pytest
from ase.build import molecule

from gdpx.providers import CapabilityKind, MaterializationError, RuntimeConfig, get_provider_manager
from gdpx.providers.allegro.manager import AllegroManager
from gdpx.providers.nequip.manager import NequipManager


@pytest.mark.parametrize('provider,newton', [('nequip', 'off'), ('allegro', 'on')])
@pytest.mark.parametrize('executor', ['ase', 'lammps'])
def test_pair_style_and_executor_routing(provider, newton, executor, tmp_path):
    model = tmp_path/'model.pth'
    model.touch()
    potential = {'provider': provider, 'parameters': {'model': str(model), 'command': 'lmp'}}
    if provider == 'nequip' or executor == 'ase':
        potential['backend'] = 'lammps'
    runtime = get_provider_manager().resolve_runtime({
        'potential': potential, 'executor': {'provider': executor, 'method': 'min'},
    })
    calc = runtime.materialization.calculator
    assert calc.pair_style == provider
    assert calc.newton == newton
    assert calc.units == 'metal'
    assert calc.atom_style == 'atomic'
    assert runtime.potential.backend == 'lammps'
    assert runtime.potential.provider == provider
    assert RuntimeConfig.from_mapping(runtime.config.to_dict()) == runtime.config
    if executor == 'ase':
        assert calc.task == 'spc'
        assert calc._write_simulation_tasks().strip() == 'run             0'
    else:
        assert runtime.materialization.metadata['provider'] == provider
        assert runtime.materialization.commands[0] == f'pair_style {provider}'
    atoms = molecule('H2O'); atoms.center(vacuum=5); atoms.pbc = True
    calc.type_list = ['H', 'O']
    calc.directory = str(tmp_path / f'{provider}-{executor}')
    calc.write_input(atoms)
    text = (tmp_path / f'{provider}-{executor}' / 'in.lammps').read_text()
    assert f'pair_style  {provider}' in text
    assert f'pair_coeff  * * {model} H O' in text
    assert f'newton  {newton}' in text


@pytest.mark.parametrize('manager_cls', [NequipManager, AllegroManager])
def test_separate_managers_preserve_input_and_canonicalize_models(manager_cls, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'model.pth').touch()
    params = {'backend': 'lammps', 'model': './model.pth', 'type_list': ['H','O']}
    original = copy.deepcopy(params)
    manager = manager_cls()
    manager.register_calculator(params)
    assert params == original
    assert manager.calc_params['model'] == [str(tmp_path/'model.pth')]
    assert manager.calc.pair_style == manager.name


@pytest.mark.parametrize('backend', ['ase', 'lammps'])
def test_old_allegro_flavour_has_migration_message(backend):
    with pytest.raises(ValueError, match='potential.provider: allegro'):
        NequipManager().register_calculator({'backend': backend, 'flavour': 'allegro'})


def test_allegro_rejects_ase_backend_for_native_lammps():
    with pytest.raises(MaterializationError, match='supported backends: lammps'):
        get_provider_manager().resolve_runtime({
            'potential': {'provider': 'allegro', 'backend': 'ase'},
            'executor': {'provider': 'lammps', 'method': 'spc'},
        })


def test_allegro_requires_model():
    with pytest.raises(ValueError, match='requires an exported model'):
        AllegroManager().register_calculator({})


def test_nequip_trainer_remains_available():
    assert get_provider_manager().require('nequip', CapabilityKind.TRAINER, 'default')


@pytest.fixture
def compiled_loader(monkeypatch):
    import sys
    import types
    from ase.calculators.emt import EMT

    calls = []
    def load(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return EMT()
    module = types.ModuleType('nequip.integrations.ase')
    module.NequIPCalculator = types.SimpleNamespace(from_compiled_model=load)
    monkeypatch.setitem(sys.modules, 'nequip.integrations.ase', module)
    return calls


@pytest.mark.parametrize('backend', [None, 'ase'])
def test_allegro_direct_ase_default_and_options(tmp_path, compiled_loader, backend):
    from ase import Atoms
    import numpy as np

    model = tmp_path/'allegro.nequip.pt2'
    model.touch()
    potential = {'provider': 'allegro', 'parameters': {
        'model': str(model), 'type_list': ['Cu'], 'device': 'cuda',
        'neighborlist_backend': 'ase', 'energy_units_to_eV': 2.0,
    }}
    if backend is not None:
        potential['backend'] = backend
    original = copy.deepcopy(potential)
    runtime = get_provider_manager().resolve_runtime({
        'potential': potential, 'executor': {'provider': 'ase', 'method': 'spc'},
    })
    assert potential == original
    assert compiled_loader == [{
        'compile_path': str(model), 'device': 'cuda',
        'chemical_species_to_atom_type_map': {'Cu': 'Cu'},
        'neighborlist_backend': 'ase', 'energy_units_to_eV': 2.0,
    }]
    assert runtime.potential.backend == 'ase'
    assert runtime.config.to_dict()['potential']['backend'] == 'ase'
    atoms = Atoms('Cu2', positions=[[0,0,0],[2.4,0,0]])
    atoms.calc = runtime.materialization.calculator
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


@pytest.mark.parametrize('committee', [False, True])
def test_allegro_ase_multiple_models_and_explicit_mapping(tmp_path, compiled_loader, committee):
    models = [tmp_path/'first.pt2', tmp_path/'second.pt2']
    for path in models:
        path.touch()
    manager = AllegroManager()
    manager.register_calculator({
        'model': [str(path) for path in models], 'estimate_uncertainty': committee,
        'type_list': ['Cu'], 'chemical_species_to_atom_type_map': {'Cu':'copper'},
    })
    assert len(compiled_loader) == (2 if committee else 1)
    for params in compiled_loader:
        assert params['device'] == 'cpu'
        assert params['chemical_species_to_atom_type_map'] == {'Cu':'copper'}
        assert 'estimate_uncertainty' not in params
        assert 'type_list' not in params
    if committee:
        assert len(manager.calc.mixer.calcs) == 2


def test_allegro_ase_missing_dependency(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'nequip.integrations.ase', None)
    model = tmp_path/'model.pt2'; model.touch()
    with pytest.raises(ModuleNotFoundError, match='compiled-model API'):
        AllegroManager().register_calculator({'model': str(model)})


def test_allegro_compiled_model_errors_are_not_hidden(tmp_path, compiled_loader, monkeypatch):
    import sys
    def broken(**kwargs):
        raise ValueError('model target does not match ASE')
    monkeypatch.setattr(sys.modules['nequip.integrations.ase'].NequIPCalculator, 'from_compiled_model', broken)
    model = tmp_path/'bad.pt2'; model.touch()
    with pytest.raises(ValueError, match='model target does not match ASE'):
        AllegroManager().register_calculator({'model': str(model)})
