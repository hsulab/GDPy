"""Candidate progress stays out of shared batch output, including inline children."""
import logging

import pytest

from gdpx import config
from gdpx.core.output import Box, candidate_logging, quiet_logging
from gdpx.execution.lifecycle import prepare_compute
from gdpx.execution.lifecycle.service import orchestrate_compute
from test_sampling_config import sampling_config
from test_sampling_lifecycle import copper


@pytest.mark.parametrize('method', ['monte_carlo', 'biased_monte_carlo', 'hybrid_monte_carlo'])
def test_batch_progress_is_written_per_candidate(tmp_path, caplog, method):
    runtime = sampling_config(method)
    runtime['executor']['parameters']['stop']['steps'] = 1
    plan = prepare_compute(runtime, [copper(), copper()], tmp_path)
    shared = logging.FileHandler(tmp_path / 'gdp.out')
    config.logger.addHandler(shared)
    config.logger.addHandler(caplog.handler)
    try:
        assert orchestrate_compute(plan).state == 'finished'
        config.logger.info('batch complete')
    finally:
        config.logger.removeHandler(shared)
        config.logger.removeHandler(caplog.handler)
        shared.close()
    for index in (0, 1):
        text = (tmp_path / f'cand{index}/mc.out').read_text()
        assert 'decision:' in text
        assert 'energy [eV]' in text
        assert 'batch complete' not in text
        if method == 'hybrid_monte_carlo':
            assert 'hybrid monte carlo | intro' in text
    for text in (caplog.text, (tmp_path / 'gdp.out').read_text()):
        assert 'decision:' not in text
        assert 'current energy [eV]' not in text
        assert 'hybrid monte carlo | intro' not in text
        assert 'batch complete' in text


@pytest.mark.parametrize('error', [RuntimeError('failed evaluation'), KeyboardInterrupt()])
def test_candidate_log_appends_and_restores_after_failure(tmp_path, caplog, error):
    original_filters = list(config.logger.filters)
    config.logger.addHandler(caplog.handler)
    try:
        outer = Box('batch')
        with outer.as_parent(), quiet_logging():
            with candidate_logging(tmp_path):
                config.logger.info('first invocation')
            with pytest.raises(type(error)):
                with candidate_logging(tmp_path):
                    with candidate_logging(tmp_path / 'inline-child'):
                        Box('candidate').line('second invocation')
                        raise error
            outer.line('batch resumed')
        assert config.logger.filters == original_filters
        assert 'batch resumed' in caplog.text
        assert 'invocation' not in caplog.text
        text = (tmp_path / 'mc.out').read_text()
        assert 'first invocation' in text and 'second invocation' in text
        assert type(error).__name__ in text
        assert not (tmp_path / 'inline-child/mc.out').exists()
    finally:
        config.logger.removeHandler(caplog.handler)


def test_concurrent_candidate_logs_are_isolated(tmp_path, caplog):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(3)
    original_filters = list(config.logger.filters)

    def run_candidate(index):
        with candidate_logging(tmp_path / f'cand{index}'):
            barrier.wait(timeout=10)
            config.logger.info(f'candidate {index} progress')
            barrier.wait(timeout=10)

    config.logger.addHandler(caplog.handler)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_candidate, index) for index in (0, 1)]
            barrier.wait(timeout=10)
            config.logger.info('concurrent batch status')
            barrier.wait(timeout=10)
            for future in futures:
                future.result(timeout=10)
        assert config.logger.filters == original_filters
        assert 'concurrent batch status' in caplog.text
        assert 'candidate 0 progress' not in caplog.text
        assert 'candidate 1 progress' not in caplog.text
        for index in (0, 1):
            text = (tmp_path / f'cand{index}/mc.out').read_text()
            assert f'candidate {index} progress' in text
            assert f'candidate {1 - index} progress' not in text
            assert 'concurrent batch status' not in text
    finally:
        config.logger.removeHandler(caplog.handler)
