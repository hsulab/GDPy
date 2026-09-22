"""Exploration framing follows actual invocation and completion boundaries."""

import io
import logging
import sys
from contextlib import closing
from types import SimpleNamespace

import pytest

from gdpx import config
from gdpx.cli import explore
from gdpx.exploration.output import exploration_output


@pytest.fixture
def output(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(config.formatter)
    monkeypatch.setattr(config.logger, 'handlers', [handler])
    old_level = config.logger.level
    config.logger.setLevel(logging.INFO)
    monkeypatch.setattr(config, 'NJOBS', 1)
    yield stream
    config.logger.setLevel(old_level)


def assert_pair(text, status):
    assert text.count('─ exploration ') == 2
    assert text.count(f'status: {status}') == 1
    assert text.count('elapsed this invocation:') == 1
    assert 'exploration |' not in text
    assert 'Exploration finished...' not in text


class Exploration:
    def __init__(self, completes_after=1, error=None):
        self.runs = 0
        self.completes_after = completes_after
        self.error = error

    def run(self):
        self.runs += 1
        if self.error is not None:
            raise self.error

    def read_convergence(self):
        return self.runs >= self.completes_after

    def as_dict(self):
        return {}


@pytest.mark.parametrize('spawn,complete,status', [(None, True, 'complete'), ('7', True, 'complete'),
                                                 ('7', False, 'waiting')])
def test_direct_and_spawned_runs(tmp_path, monkeypatch, output, spawn, complete, status):
    exploration = Exploration(completes_after=1 if complete else 2)
    monkeypatch.setattr(explore, 'create_exploration', lambda params: exploration)
    explore.run_exploration({}, runtime={}, directory=tmp_path, spawn=spawn,
                           input_path='explore.yaml', random_seed=12345)
    text = output.getvalue()
    assert_pair(text, status)
    assert f'explorations: {int(complete)} complete | {int(not complete)} pending' in text
    assert 'input: explore.yaml' in text
    assert 'random seed: 12345' in text
    assert exploration.directory == tmp_path
    assert exploration.runs == 1


@pytest.mark.parametrize('debug', [False, True])
def test_worker_diagnostics_require_debug(tmp_path, monkeypatch, output, debug):
    class ReportingExploration(Exploration):
        def run(self):
            super().run()
            config._print('algorithm progress')
            config.logger.warning('algorithm warning')

    config.logger.setLevel(logging.DEBUG if debug else logging.INFO)
    exploration = ReportingExploration()
    monkeypatch.setattr(explore, 'create_exploration', lambda params: exploration)
    explore.run_exploration({}, runtime={}, directory=tmp_path)
    text = output.getvalue()
    assert_pair(text, 'complete')
    for diagnostic in ('<<-- ExplorationBasedWorker+run -->>', '.: direct',
                       'exp_index=0', 'progress: 1/1'):
        assert (f'DEBUG: {diagnostic}' in text) == debug
        assert f'INFO: {diagnostic}' not in text
    assert 'INFO: algorithm progress' in text
    assert 'WARNING: algorithm warning' in text

    # Inspecting an already completed run should not add submission chatter.
    output.seek(0)
    output.truncate()
    explore.run_exploration({}, runtime={}, directory=tmp_path)
    text = output.getvalue()
    assert_pair(text, 'complete')
    submitted = [line for line in text.splitlines() if 'was submitted.' in line]
    assert len(submitted) == int(debug)
    assert all('DEBUG:' in line for line in submitted)
    assert exploration.runs == 1


@pytest.mark.parametrize('finished', [[], [0], [0, 1]])
def test_scheduler_completion_counts_without_extra_work(tmp_path, monkeypatch, output, finished):
    calls = []
    worker = SimpleNamespace(
        run=lambda: calls.append('run'),
        inspect=lambda **kw: calls.append(('inspect', kw)),
        job_store=SimpleNamespace(get_finished=lambda: [SimpleNamespace(group_number=i) for i in finished]),
    )
    monkeypatch.setattr(explore, 'create_exploration', lambda params: [Exploration(), Exploration()])
    monkeypatch.setattr(explore, 'ExplorationBasedWorker', lambda **kw: worker)
    explore.run_exploration({}, runtime={}, directory=tmp_path)
    text = output.getvalue()
    assert_pair(text, 'complete' if len(finished) == 2 else 'waiting')
    assert f'explorations: {len(finished)} complete | {2-len(finished)} pending' in text
    assert calls == ['run', ('inspect', {'resubmit': True})]


def test_multiple_spawned_directories_and_polling(tmp_path, monkeypatch, output):
    from gdpx.execution.workers import explore as worker_module
    explorations = [Exploration(3), Exploration(2)]
    sleeps = []
    monkeypatch.setattr(worker_module.time, 'sleep', sleeps.append)
    monkeypatch.setattr(explore, 'create_exploration', lambda params: explorations)
    explore.run_exploration({}, runtime={}, directory=tmp_path, spawn='2,5', wait=0.01)
    assert_pair(output.getvalue(), 'complete')
    assert [exp.directory for exp in explorations] == [tmp_path / 'expo.02', tmp_path / 'expo.05']
    assert [exp.runs for exp in explorations] == [3, 2]
    assert sleeps == [0.01] * 3


@pytest.mark.parametrize('phase', ['setup', 'run'])
@pytest.mark.parametrize('error', [RuntimeError('calculation failed'), KeyboardInterrupt()])
def test_failures_preserve_exception_and_reset_context(tmp_path, monkeypatch, output, phase, error):
    def create(params):
        if phase == 'setup':
            raise error
        return Exploration(error=error)
    monkeypatch.setattr(explore, 'create_exploration', create)
    with pytest.raises(type(error)) as caught:
        explore.run_exploration({}, runtime={}, directory=tmp_path, spawn='0')
    assert caught.value is error
    assert_pair(output.getvalue(), 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
    assert 'complete |' not in output.getvalue()
    output.seek(0)
    output.truncate()
    monkeypatch.setattr(explore, 'create_exploration', lambda params: Exploration())
    explore.run_exploration({}, runtime={}, directory=tmp_path, spawn='0')
    assert_pair(output.getvalue(), 'complete')


def test_ascii_paths_elapsed_and_file_logging(tmp_path, monkeypatch):
    from gdpx.exploration import output as module
    stream = io.StringIO()
    path = tmp_path / 'gdp.out'
    now = [10.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    with closing(logging.FileHandler(path, encoding='ascii')) as file_handler:
        handlers = [logging.StreamHandler(stream), file_handler]
        for handler in handlers:
            handler.setFormatter(config.formatter)
        monkeypatch.setattr(config.logger, 'handlers', handlers)
        with exploration_output(tmp_path / ('long-path-' * 20), random_seed=0) as report:
            report.start(1, 'direct')
            report.pending = 0
            now[0] = 12.5
    text = stream.getvalue()
    assert text == path.read_text()
    assert text.isascii()
    assert text.count('+- exploration ') == 2
    assert 'random seed: 0' in text
    assert 'elapsed this invocation: 2.5 s' in text
    assert all(len(line.split(' - INFO: ')[1]) == 76 for line in text.splitlines())


@pytest.mark.parametrize('debug', [False, True])
@pytest.mark.parametrize('invalid', [False, True])
def test_cli_framing_and_random_state(tmp_path, monkeypatch, output, debug, invalid):
    from gdpx import main
    monkeypatch.setattr(main, 'bootstrap_registries', lambda **kw: None)
    monkeypatch.setattr(config, 'GRNG', config.GRNG)
    monkeypatch.setattr(explore, 'create_exploration', lambda params: Exploration())
    source = tmp_path / 'explore.yaml'
    source.write_text('runtime: {}\n' if not invalid else '[')
    args = ['gdp', '-d', str(tmp_path), '--log', '', '-rs', '12345']
    if debug:
        args.append('--debug')
    monkeypatch.setattr(sys, 'argv', args + ['explore', str(source), '--spawn', '0'])
    if invalid:
        with pytest.raises(Exception):
            main.main()
    else:
        main.main()
    text = output.getvalue()
    assert_pair(text, 'failed' if invalid else 'complete')
    assert 'random seed: 12345' in text
    assert config.LOGO_LINES[0] not in text
    assert text.count('GLOBAL RANDOM SEED : 12345') == (2 if debug else 0)
    assert ('bit_generator' in text) == debug


def test_other_commands_keep_boilerplate(tmp_path, monkeypatch, output):
    from gdpx import main
    monkeypatch.setattr(main, 'bootstrap_registries', lambda **kw: None)
    monkeypatch.setattr(config, 'GRNG', config.GRNG)
    monkeypatch.setattr(sys, 'argv', ['gdp', '-d', str(tmp_path), '--log', '', '-rs', '12'])
    main.main()
    text = output.getvalue()
    assert config.LOGO_LINES[0] in text
    assert text.count('GLOBAL RANDOM SEED : 12') == 2
    assert text.count('bit_generator') == 2
    assert '─ exploration ' not in text
