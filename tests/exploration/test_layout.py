"""Exploration layout, restart identity, and scheduler path integration."""
import json
import pathlib
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from tinydb import TinyDB

from gdpx.exploration.layout import expedition_directories, exploration_layout
from gdpx.execution.workers.explore import ExpeditionBasedWorker
from gdpx.execution.schedulers.direct import DirectScheduler


class Expedition:
    def __init__(self, complete=True):
        self.complete = complete
        self.runs = 0

    def as_dict(self):
        return {'method': 'test', 'runtime': {}}

    def run(self):
        self.runs += 1
        if self.complete:
            (self.directory / 'done').write_text('complete')

    def read_convergence(self):
        return (self.directory / 'done').exists()

    def get_workers(self):
        return [self.directory]


class Queue(DirectScheduler):
    name = 'queue'

    def __init__(self):
        super().__init__()
        self.calls = []
        self.finished = False

    def submit(self, func_to_execute=None):
        self.calls.append((self.job_name, self.script, self.user_commands))
        return str(len(self.calls))

    def is_finished(self):
        return self.finished


@pytest.mark.parametrize('count,width', [(2, 2), (9, 2), (10, 2), (99, 2), (100, 4),
                                       (9999, 4), (10000, 6)])
def test_even_width(count, width):
    names = expedition_directories(count)
    assert names[0] == 'expo.' + '0' * width
    assert names[-1] == f'expo.{count-1:0{width}d}'
    assert len(names) == count


@pytest.mark.parametrize('count', [1, 2])
def test_layout_resume_and_retrieval(tmp_path, count):
    worker = ExpeditionBasedWorker([Expedition() for _ in range(count)], DirectScheduler(), directory=tmp_path)
    worker.run()
    worker.inspect()
    names = expedition_directories(count)
    assert (tmp_path / '_meta' / '_scheduler.json').exists()
    assert not (tmp_path / '_direct_jobs.json').exists()
    assert len(list((tmp_path / '_meta').glob('exp-*.json'))) == count
    assert len(list((tmp_path / '_meta').glob('run.script-*'))) == count
    for name in names:
        assert (tmp_path / name / 'done').exists()
        assert not (tmp_path / name / '_data').exists()
        if name != '.':
            assert not (tmp_path / name / '_meta').exists()
    restarted = ExpeditionBasedWorker([Expedition() for _ in range(count)], DirectScheduler(), directory=tmp_path)
    restarted.run()
    assert all(exp.runs == 0 for exp in restarted.expeditions)
    assert restarted.retrieve() == [(tmp_path / name).resolve() for name in names]
    assert restarted.retrieve() == []
    assert restarted.retrieve(include_retrieved=True) == [(tmp_path / name).resolve() for name in names]
    before = (tmp_path / '_meta' / '_scheduler.json').read_bytes()
    changed = ExpeditionBasedWorker([Expedition() for _ in range(count+1)], DirectScheduler(), directory=tmp_path)
    with pytest.raises(ValueError, match='count changed'):
        changed.run()
    assert (tmp_path / '_meta' / '_scheduler.json').read_bytes() == before


@pytest.mark.parametrize('legacy', ['_direct_jobs.json', 'expedition-0'])
def test_legacy_rejected_without_metadata(tmp_path, legacy):
    path = tmp_path / legacy
    if path.suffix:
        path.write_text('{}')
    else:
        path.mkdir()
    worker = ExpeditionBasedWorker(Expedition(), DirectScheduler(), directory=tmp_path)
    with pytest.raises(RuntimeError, match='Legacy exploration layout'):
        worker.run()
    assert not (tmp_path / '_meta').exists()
    assert path.exists()


def test_shuffled_records_and_resubmit_after_restart(tmp_path):
    scheduler = Queue()
    worker = ExpeditionBasedWorker([Expedition(), Expedition()], scheduler, directory=tmp_path)
    worker.run()
    store = worker.job_store.path
    with TinyDB(store) as db:
        records = list(db.all())
        db.truncate()
        db.insert_multiple(reversed(records))
    restarted_scheduler = Queue()
    restarted_scheduler.finished = True
    restarted = ExpeditionBasedWorker([Expedition(), Expedition()], restarted_scheduler, directory=tmp_path)
    restarted.run()
    assert not restarted_scheduler.calls
    restarted.inspect(resubmit=True)
    assert len(restarted_scheduler.calls) == 2
    for job_name, script, command in restarted_scheduler.calls:
        record = next(record for record in records if record['gdir'] == job_name)
        assert f'--spawn {record["group_number"]}' in command
        assert record['wdir_names'][0] in command
        assert script.parent == tmp_path / '_meta'
        assert record['uid'] in script.name
        assert command in script.read_text()
    assert all(record.attempt == 2 for record in restarted.job_store.get_running())
    for name in expedition_directories(2):
        (tmp_path / name / 'done').write_text('done')
    restarted.inspect()
    assert set(restarted.retrieve()) == {tmp_path / 'expo.00', tmp_path / 'expo.01'}


@pytest.mark.parametrize('count', [1, 2])
def test_generated_script_launch_directory_and_input(tmp_path, count):
    root = tmp_path / 'run with spaces'
    scheduler = Queue()
    scheduler.environs = 'gdp() { pwd > launch-cwd; printf "%s\\n" "$@" > launch-args; }'
    worker = ExpeditionBasedWorker([Expedition() for _ in range(count)], scheduler, directory=root)
    worker.run()
    for _, script, _ in scheduler.calls:
        subprocess.run(['bash', str(script)], cwd=script.parent, check=True)
    for index, name in enumerate(expedition_directories(count)):
        wdir = root / name
        assert (wdir / 'launch-cwd').read_text().strip() == str(wdir.resolve())
        args = (wdir / 'launch-args').read_text().splitlines()
        assert args[0] == 'explore'
        assert (wdir / args[1]).is_file()
        assert args[-2:] == ['--spawn', str(index)]
        assert json.loads((wdir / args[1]).read_text())['runtime'] == {}


@pytest.mark.parametrize('name', ['.', 'expo.00'])
def test_ssh_selective_sync_preserves_metadata_and_siblings(tmp_path, name):
    from gdpx.execution.schedulers.remote import SshTransport
    local = tmp_path / 'local'
    remote_root = tmp_path / 'remote' / 'job'
    for root in (local, remote_root):
        (root / '_meta').mkdir(parents=True)
        (root / 'expo.01').mkdir()
        (root / 'expo.01' / 'result').write_text('sibling')
        (root / name).mkdir(exist_ok=True)
    (local / '_meta' / '_scheduler.json').write_text('local metadata')
    (remote_root / '_meta' / '_scheduler.json').write_text('stale metadata')
    (local / name / 'result').write_text('old')
    (local / name / 'obsolete').write_text('old')
    (remote_root / name / 'result').write_text('new remote result')
    if name != '.':
        (remote_root / 'expo.01' / 'result').write_text('wrong sibling result')
    # Calculation-worker metadata below the output root must still synchronize.
    nested = pathlib.Path(name) / 'calculations' / '_meta'
    (remote_root / nested).mkdir(parents=True)
    (remote_root / nested / 'state').write_text('nested metadata')

    class Sftp:
        def listdir_attr(self, directory):
            return [SimpleNamespace(filename=p.name, st_mode=p.stat().st_mode)
                    for p in pathlib.Path(directory).iterdir()]
        def lstat(self, path):
            return pathlib.Path(path).stat()
        stat = lstat
        def get(self, source, destination):
            shutil.copyfile(source, destination)
        def close(self):
            pass

    client = SimpleNamespace(open_sftp=lambda: Sftp(), close=lambda: None)
    transport = SshTransport(DirectScheduler(), 'test', str(remote_root.parent))
    transport.local_root = local
    transport.script = local / '_meta' / 'run.script'
    transport.job_name = 'job'
    transport._client = lambda: client
    transport.sync([name], root_relative=True)
    assert (local / name / 'result').read_text() == 'new remote result'
    assert not (local / name / 'obsolete').exists()
    assert (local / '_meta' / '_scheduler.json').read_text() == 'local metadata'
    assert (local / 'expo.01' / 'result').read_text() == 'sibling'
    assert (local / nested / 'state').read_text() == 'nested metadata'


@pytest.mark.parametrize('count', [1, 2])
def test_active_workflow_uses_previous_layout(tmp_path, count):
    from gdpx.workflow.nodes.expedition import explore
    from gdpx.workflow.session.variable import Variable
    previous = tmp_path / 'iter.0000' / 'search'
    current = tmp_path / 'iter.0001' / 'search'
    names = exploration_layout(previous, count, create=True)
    updates = []
    expeditions = [Expedition() for _ in range(count)]
    for expedition in expeditions:
        expedition.update_active_params = updates.append
    operation = explore(Variable(expeditions), active=True, directory=current)
    results = operation.forward(expeditions, None, DirectScheduler())
    assert updates == [previous / name for name in names]
    assert results == [(current / name).resolve() for name in names]
    assert operation.status == 'finished'


def test_multi_spawn_uses_saved_padding(tmp_path, monkeypatch):
    from gdpx.cli import explore
    exploration_layout(tmp_path, 100, create=True)
    expeditions = [Expedition(), Expedition()]
    # The CLI normally creates these directories through worker submission.
    for index in (2, 5):
        (tmp_path / f'expo.{index:04d}').mkdir()
    monkeypatch.setattr(explore, 'create_expedition', lambda params: expeditions)
    explore.run_expedition({}, runtime={}, directory=tmp_path, spawn='2,5')
    assert [exp.directory for exp in expeditions] == [tmp_path / 'expo.0002', tmp_path / 'expo.0005']
    assert not (tmp_path / 'expo.02').exists()


def test_ssh_stages_shared_input_and_nested_job_records(tmp_path):
    from gdpx.execution.schedulers.remote import SshTransport
    transport = SshTransport(DirectScheduler(), 'test', '/scratch')
    transport.local_root = tmp_path
    metadata = tmp_path / '_meta'
    metadata.mkdir()
    records = metadata / '_scheduler.json'
    records.write_text('parent jobs')
    (metadata / 'exp-input.json').write_text('input')
    nested = tmp_path / 'expo.00' / 'calculations'
    nested.mkdir(parents=True)
    (nested / '_direct_jobs.json').write_text('calculation jobs')
    transport.staging_excludes = {records.resolve()}
    uploaded = []
    sftp = SimpleNamespace(stat=lambda path: None, put=lambda local, remote: uploaded.append(pathlib.Path(local)))
    transport._transfer(sftp, tmp_path, pathlib.PurePosixPath('/scratch/job'))
    assert records not in uploaded
    assert metadata / 'exp-input.json' in uploaded
    assert nested / '_direct_jobs.json' in uploaded


def test_pbs_script_recovers_submission_directory(tmp_path):
    import os
    from gdpx.execution.schedulers.pbs import PbsScheduler
    root = tmp_path / 'run with spaces'
    scheduler = PbsScheduler(is_dry_run=True)
    scheduler.environs = 'gdp() { pwd > launch-cwd; }'
    worker = ExpeditionBasedWorker(Expedition(), scheduler, directory=root)
    worker.run()
    environment = dict(os.environ, PBS_O_WORKDIR=str(root / '_meta'))
    subprocess.run(['bash', str(scheduler.script)], cwd=tmp_path, env=environment, check=True)
    assert (root / 'launch-cwd').read_text().strip() == str(root.resolve())


def test_previous_manifest_name_preserves_restart_identity(tmp_path):
    metadata = tmp_path / '_meta'
    metadata.mkdir()
    previous = metadata / 'exploration.json'
    content = json.dumps({'version': 1, 'count': 2, 'directories': ['expo.00', 'expo.01']})
    previous.write_text(content)
    assert exploration_layout(tmp_path) == ['expo.00', 'expo.01']
    assert previous.exists()  # Read-only resolution does not migrate files.
    with pytest.raises(ValueError, match='count changed'):
        exploration_layout(tmp_path, 1, create=True)
    assert previous.read_text() == content
    assert exploration_layout(tmp_path, 2, create=True) == ['expo.00', 'expo.01']
    assert json.loads((metadata / '_scheduler.json').read_text())['layout']['1'] == json.loads(content)
    assert not (metadata / 'layout.json').exists()
    assert not previous.exists()


@pytest.mark.parametrize('manifest_name', ['layout.json', 'exploration.json'])
def test_consolidate_scheduler_records_without_resubmission(tmp_path, manifest_name):
    metadata = tmp_path / '_meta'
    metadata.mkdir()
    layout = {'version': 1, 'count': 1, 'directories': ['.']}
    (metadata / manifest_name).write_text(json.dumps(layout))
    old_store = metadata / '_queue_jobs.json'
    record = dict(uid='original', gdir='original-job', group_number=0,
                  wdir_names=['.'], queued=True, finished=True, retrieved=True,
                  scheduler_job_id='123', attempt=3)
    original = {'_default': {'7': record}}
    old_store.write_text(json.dumps(original))
    scheduler = Queue()
    worker = ExpeditionBasedWorker(Expedition(), scheduler, directory=tmp_path)
    with pytest.raises(ValueError, match='count changed'):
        exploration_layout(tmp_path, 2, create=True, scheduler='queue')
    assert old_store.exists() and (metadata / manifest_name).exists()
    worker.run()
    assert not scheduler.calls
    assert worker.job_store.path == metadata / '_scheduler.json'
    saved = json.loads(worker.job_store.path.read_text())
    assert saved == dict(original, layout={'1': layout}, scheduler={'1': {'provider': 'queue'}})
    assert len(worker.job_store) == 1
    assert worker.job_store.get_retrieved()[0].doc_id == 7
    assert worker.job_store.get_retrieved()[0].attempt == 3
    assert worker.get_number_of_running_jobs() == 0
    worker.job_store.mark_submitted('original-job', '456')
    assert json.loads(worker.job_store.path.read_text())['layout']['1'] == layout
    assert exploration_layout(tmp_path) == ['.']
    assert not old_store.exists()
    assert not (metadata / manifest_name).exists()
    assert not (metadata / '_queue_jobs.json').exists()


@pytest.mark.parametrize('provider', ['direct', 'queue'])
def test_provider_specific_database_migrates_and_rejects_provider_change(tmp_path, provider):
    metadata = tmp_path / '_meta'
    metadata.mkdir()
    layout = {'version': 1, 'count': 1, 'directories': ['.']}
    record = {'uid': 'same', 'gdir': 'same-job', 'queued': True, 'group_number': 0,
              'wdir_names': ['.'], 'scheduler_job_id': '42', 'attempt': 2}
    original = {'layout': {'1': layout}, '_default': {'5': record}}
    previous = metadata / f'_{provider}.json'
    previous.write_text(json.dumps(original))
    other = 'queue' if provider == 'direct' else 'direct'
    with pytest.raises(ValueError, match='provider changed'):
        exploration_layout(tmp_path, 1, create=True, scheduler=other)
    assert previous.exists()
    assert not (metadata / '_scheduler.json').exists()
    exploration_layout(tmp_path, 1, create=True, scheduler=provider)
    target = metadata / '_scheduler.json'
    saved = json.loads(target.read_text())
    assert saved == dict(original, scheduler={'1': {'provider': provider}})
    assert not previous.exists()
    before = target.read_bytes()
    with pytest.raises(ValueError, match='provider changed'):
        exploration_layout(tmp_path, 1, create=True, scheduler=other)
    assert target.read_bytes() == before
    assert exploration_layout(tmp_path) == ['.']


def test_conflicting_provider_files_are_not_combined(tmp_path):
    metadata = tmp_path / '_meta'
    metadata.mkdir()
    data = {'layout': {'1': {'version': 1, 'count': 1, 'directories': ['.']}}, '_default': {}}
    for provider in ('direct', 'slurm'):
        (metadata / f'_{provider}.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Conflicting scheduler providers'):
        exploration_layout(tmp_path, 1, create=True)
    assert not (metadata / '_scheduler.json').exists()
    assert len(list(metadata.iterdir())) == 2
