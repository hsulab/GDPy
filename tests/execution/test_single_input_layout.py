"""Single-task workers share their directory with calculation outputs safely."""
from ase import Atoms
import pytest

from gdpx.execution.factory import create_worker
from gdpx.execution.lifecycle import prepare_compute, collect_compute
from gdpx.execution.lifecycle.service import orchestrate_compute


def runtime():
    return {'potential': {'provider': 'emt'},
            'executor': {'provider': 'ase', 'method': 'spc'}}


@pytest.mark.parametrize('count', [1, 2])
def test_layout_collection_restart_and_archive(tmp_path, count):
    inputs = [Atoms('Cu', positions=[[i, 0, 0]]) for i in range(count)]
    plan = prepare_compute(runtime(), inputs, tmp_path)
    assert orchestrate_compute(plan).state == 'finished'
    names = ['.'] if count == 1 else ['cand0', 'cand1']
    for name in names:
        assert (tmp_path / name / 'traj.xyz').is_file()
    if count == 1:
        assert not (tmp_path / 'cand0').exists()
    assert orchestrate_compute(plan).state == 'finished'
    assert collect_compute(plan, archive=True).number_of_trajectories == count
    assert (tmp_path / '_meta/manifest.json').is_file()
    assert (tmp_path / '_meta/status.json').is_file()
    assert collect_compute(plan).number_of_trajectories == count


def test_multiple_variants_still_have_separate_directories(tmp_path):
    config = runtime()
    config['executor']['broadcast'] = {'random_seed': [1, 2]}
    plan = prepare_compute(config, [Atoms('Cu')], tmp_path)
    assert orchestrate_compute(plan).state == 'finished'
    assert (tmp_path / 'cand0/traj.xyz').is_file()
    assert (tmp_path / 'cand1/traj.xyz').is_file()


def test_single_worker_filters_root_results(tmp_path):
    config = runtime()
    config['dispatch'] = {'worker': 'single'}
    worker = create_worker(config, directory=tmp_path)
    worker.run([Atoms('Cu')])
    worker.inspect()
    frames = worker.retrieve(given_wdirs=['.'])[0]
    assert frames[-1].info['wdir'] == '.'


def test_checkpoint_backup_preserves_worker_metadata(tmp_path):
    worker = create_worker(runtime(), directory=tmp_path)
    worker.driver.directory = tmp_path
    (tmp_path / '_meta').mkdir()
    (tmp_path / '_meta/manifest.json').write_text('inputs')
    (tmp_path / 'gdp.out').write_text('batch log')
    assert worker.driver._save_checkpoint() is None
    (tmp_path / 'traj.xyz').write_text('calculation output')
    checkpoint = worker.driver._save_checkpoint()
    assert (tmp_path / '_meta/manifest.json').read_text() == 'inputs'
    assert (tmp_path / 'gdp.out').read_text() == 'batch log'
    assert (checkpoint / 'traj.xyz').read_text() == 'calculation output'


def test_root_archive_remains_readable_after_moving_run(tmp_path):
    import shutil

    source = tmp_path / 'source'
    plan = prepare_compute(runtime(), [Atoms('Cu')], source)
    assert orchestrate_compute(plan).state == 'finished'
    collect_compute(plan, archive=True)
    moved = tmp_path / 'moved'
    shutil.copytree(source, moved)
    assert collect_compute(moved).number_of_trajectories == 1
