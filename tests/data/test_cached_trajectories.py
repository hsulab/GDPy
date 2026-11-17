import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write, read
from gdpx.data.trajectories import read_cached_trajectories
from gdpx.utils.archive import create_zstd_archive


def save(path, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Atoms('H', positions=[[s*.1]*3], info={'step': s}) for s in steps]
    for frame in frames:
        frame.calc = SinglePointCalculator(frame, energy=0.0, forces=np.zeros((len(frame), 3)))
    write(path, frames)


def test_compute_exploration_archives_and_live_override(tmp_path):
    root = tmp_path/'explore'
    gen = root/'tmp_folder/gen0'; gen.mkdir(parents=True)
    stage = tmp_path/'stage'
    save(stage/'cand2/0000.run/traj.xyz', [0, 1, 2, 3])
    save(stage/'cand2/traj.xyz', [2, 3, 4])
    save(stage/'cand10/traj.xyz', [0, 1])
    create_zstd_archive(gen/'cand.tar.zst', [(stage/'cand10', 'cand10'), (stage/'cand2', 'cand2')])
    # A resumed live trajectory supersedes the same archived segment.
    save(gen/'cand2/traj.xyz', [2, 3, 4, 5])
    save(root/'tmp_folder/gen1/cand0/traj.xyz', [0, 1])
    save(root/'results/end_frames.xyz', [99])
    trajectories = read_cached_trajectories(root)
    assert [[a.info['step'] for a in t] for t in trajectories] == [list(range(6)), [0, 1], [0, 1]]
    assert len({t[0].info['trajectory_id'] for t in trajectories}) == 3
    assert [a.info['trajectory_frame'] for a in trajectories[0]] == list(range(6))
    assert len(read_cached_trajectories(gen)) == 2


def test_legacy_archive_and_no_extraction(tmp_path):
    source = tmp_path/'source'; save(source/'cand0/traj.xyz', [0, 1])
    root = tmp_path/'root'; root.mkdir()
    with tarfile.open(root/'cand.tgz', 'w:gz') as tar:
        tar.add(source/'cand0', arcname='cand0')
    assert len(read_cached_trajectories(root)[0]) == 2
    assert not (root/'cand0').exists()


def test_unsafe_archive_and_empty(tmp_path):
    with pytest.raises(ValueError, match='No trajectories'):
        read_cached_trajectories(tmp_path)
    with tarfile.open(tmp_path/'cand.tgz', 'w:gz') as tar:
        entry = tarfile.TarInfo('../bad/traj.xyz'); entry.size = 1
        tar.addfile(entry, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe'):
        read_cached_trajectories(tmp_path)


def test_cli_selection_keeps_trajectories_separate(tmp_path):
    from gdpx.cli.select import run_selection
    root = tmp_path/'compute'
    save(root/'cand0/traj.xyz', [0, 1, 2, 3])
    save(root/'cand1/traj.xyz', [0, 1])
    builder = tmp_path/'source.yaml'
    builder.write_text(yaml.safe_dump({'method':'cached_trajectories', 'directory':str(root)}))
    selection = tmp_path/'select.yaml'
    selection.write_text(yaml.safe_dump({'method':'cartesian_rmse', 'rmse_interval':.15}))
    run_selection(selection, [str(builder)], directory=tmp_path/'out')
    frames = read(tmp_path/'out/selected_frames.xyz', ':')
    assert [a.info['step'] for a in frames] == [2, 3, 1]


class Reader:
    directory = None
    def read_trajectory(self, archive_path=None):
        return [Atoms('H', info={'step':0})]


def test_extract_cache_without_worker_and_legacy_worker(tmp_path):
    from gdpx.workflow.nodes.driver import extract_cache
    root = tmp_path/'compute'
    save(root/'cand0/traj.xyz', [0, 1])
    save(root/'cand1/traj.xyz', [0, 1, 2])
    op = extract_cache(cache_wdirs=[root], directory=tmp_path/'cache')
    result = op.forward()
    assert result.shape == (1, 2, 3)
    # Cache roundtrip preserves per-trajectory grouping.
    assert op.forward().shape == result.shape
    raw = tmp_path/'raw'; raw.mkdir()
    worker = SimpleNamespace(driver=Reader())
    op = extract_cache(cache_wdirs=[raw], save_h5=False, directory=tmp_path/'legacy')
    assert op.forward([worker]).shape == (1, 1)
    assert worker.driver.directory is None
    single = tmp_path/"single"
    save(single/"cand0/traj.xyz", [0, 1])
    op = extract_cache(cache_wdirs=[single, root], save_h5=False, directory=tmp_path/"multi")
    assert op.forward([worker]).shape == (2, 2, 1)
