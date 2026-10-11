import copy
from pathlib import Path

import pytest
import yaml

from gdpx.execution.schedulers.factory import canonicalise_scheduler
from gdpx.providers import ProviderConfigurationError, RuntimeConfig
from gdpx.providers.configuration import scheduler_component
from gdpx.user_config import user_config_path
from gdpx.workflow.configuration import WorkflowConfigError, load_workflow
from gdpx.workflow.state_store import workflow_fingerprint


@pytest.fixture
def presets(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    path = tmp_path / 'gdpx/config.yaml'
    path.parent.mkdir()
    data = {'schedulers': {'gpu': {
        'provider': 'slurm',
        'parameters': {'account': 'gpu_account', 'qos': 'premium', 'time': '01:00:00',
                       'environs': ['module load conda', 'conda activate gdp3']},
        'transport': {'provider': 'ssh', 'parameters': {'hostname': 'cluster', 'remote_wdir': '/scratch/jobs'}},
    }}}
    path.write_text(yaml.safe_dump(data))
    return path, data


def test_default_config_directory(tmp_path, monkeypatch):
    monkeypatch.delenv('XDG_CONFIG_HOME', raising=False)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    assert user_config_path() == tmp_path / '.config/gdpx/config.yaml'


def test_named_scheduler_deep_overrides_and_list_replacement(presets):
    path, data = presets
    original = copy.deepcopy(data)
    assert scheduler_component('gpu').to_dict() == data['schedulers']['gpu']
    config = scheduler_component({'preset': 'gpu', 'parameters': {'qos': 'regular', 'environs': [], 'time': None},
                                  'transport': {'parameters': {'hostname': 'other'}}}).to_dict()
    assert config['parameters'] == {'account': 'gpu_account', 'qos': 'regular', 'time': None, 'environs': []}
    assert config['transport']['parameters'] == {'hostname': 'other', 'remote_wdir': '/scratch/jobs'}
    assert yaml.safe_load(path.read_text()) == original
    assert 'preset' not in config


def test_runtime_and_scheduler_factory_resolve_presets(presets):
    runtime = RuntimeConfig.from_mapping({'schema_version': 4,
        'potential': {'provider': 'emt'}, 'executor': {'provider': 'ase', 'method': 'spc'},
        'scheduler': {'preset': 'gpu', 'transport': None}})
    assert runtime.to_dict()['scheduler']['parameters']['account'] == 'gpu_account'
    assert 'preset' not in runtime.to_dict()['scheduler']
    scheduler = canonicalise_scheduler({'preset': 'gpu', 'transport': None, 'parameters': {'ntasks': 4}})
    assert scheduler.name == 'slurm' and scheduler.parameters['ntasks'] == 4
    assert scheduler.environs == ['module load conda', 'conda activate gdp3']


def test_inline_scheduler_does_not_read_user_config(presets):
    path, _ = presets
    path.write_text('[invalid')
    assert scheduler_component({'provider': 'slurm', 'parameters': {'ntasks': 2}}).parameters['ntasks'] == 2


@pytest.mark.parametrize('value, message', [
    ('missing', 'available: gpu'), ({'preset': ''}, 'nonempty name'),
])
def test_invalid_preset_reference(presets, value, message):
    with pytest.raises(ProviderConfigurationError, match=message):
        scheduler_component(value)


@pytest.mark.parametrize('content, message', [
    ('[invalid', 'Cannot load scheduler preset'), ('schedulers: []', 'schedulers.*mapping'),
    ('schedulers: {gpu: {preset: other}}', 'must define a provider'),
])
def test_invalid_user_config(presets, content, message):
    path, _ = presets
    path.write_text(content)
    with pytest.raises(ProviderConfigurationError, match=message):
        scheduler_component('gpu')


def test_missing_user_config(presets):
    path, _ = presets
    path.unlink()
    with pytest.raises(ProviderConfigurationError, match='Cannot load scheduler preset.*config.yaml'):
        scheduler_component('gpu')


def test_workflow_presets_preserve_inline_fingerprint_and_snapshot(presets, tmp_path):
    from gdpx.bootstrap import bootstrap_registries
    bootstrap_registries(disable_import_info=True)
    path, data = presets
    scheduler = data['schedulers']['gpu']
    config = {'resources': {
        'queue': {'__type__': 'scheduler', 'options': scheduler},
        'runtime': {'__type__': 'runtime', 'options': {'scheduler': scheduler}},
    }, 'steps': {'assembled': {'__type__': 'assemble', 'options': {'scheduler': scheduler}}},
       'workflow': {'targets': ['assembled']}}
    workflow = tmp_path / 'active.yaml'
    workflow.write_text(yaml.safe_dump(config))
    inline = load_workflow(workflow)
    config['resources']['queue']['options'] = {'preset': 'gpu'}
    config['resources']['runtime']['options']['scheduler'] = 'gpu'
    config['steps']['assembled']['options']['scheduler'] = {'preset': 'gpu'}
    workflow.write_text(yaml.safe_dump(config))
    resolved = load_workflow(workflow)
    assert workflow_fingerprint(resolved) == workflow_fingerprint(inline)
    data['schedulers']['gpu']['parameters']['qos'] = 'regular'
    path.write_text(yaml.safe_dump(data))
    assert resolved.resources['queue'].options['parameters']['qos'] == 'premium'
    assert workflow_fingerprint(load_workflow(workflow)) != workflow_fingerprint(inline)
    path.unlink()
    with pytest.raises(WorkflowConfigError, match='resources.queue.*Cannot load scheduler preset'):
        load_workflow(workflow)
