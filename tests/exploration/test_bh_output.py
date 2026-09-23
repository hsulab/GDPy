"""Scientific progress counts and scoped logging for BH generation blocks."""
import io
import json
import logging
import re
from types import SimpleNamespace

import pytest

from gdpx import config
from gdpx.exploration.basin_hopping.output import GenerationReporter, bh_logging


class Row(dict):
    def __getattr__(self, key):
        return self[key]


def row(generation=1, step=1, accepted=True, extinct=0, target=-10.):
    return Row(generation=generation, extinct=extinct, target=target,
               data={'round': step, 'accepted': accepted})


def reporter(tmp_path, rows, generation=1, unicode=True, resumed=False):
    messages = []
    db = SimpleNamespace(connection=SimpleNamespace(select=lambda **kwargs: iter(rows)))
    output = GenerationReporter(db, tmp_path, generation, 1, 4, 2, 4,
                                'formation_energy', resumed=resumed, emit=messages.append, unicode=unicode)
    return output, messages


def journal(tmp_path, events):
    content = ''.join(json.dumps(event) + '\n' for event in events)
    (tmp_path / 'events.jsonl').write_text(content)
    return len(content.encode())


def test_committed_counts_include_rejected_extinction_and_exclude_future(tmp_path):
    rows = [row(generation=0, target=-9), row(extinct=1, target=-50),
            row(accepted=False, extinct=1, target=-30), row(accepted=False, target=-12),
            row(step=2, target=-100)]
    output, messages = reporter(tmp_path, rows)
    offset = journal(tmp_path, [dict(step=0, decisions=[0]*4), dict(step=1, decisions=[4, 1, 2, 1])])
    output.progress(1, offset)
    output.finish('WAITING', 'waiting for round 2/2 evaluations')
    text = '\n'.join(messages)
    assert '3 evaluations | 1 accepted | 2 rejected' in text
    assert '1 invalid | 2 extinct | 1 restarts' in text
    assert 'best eligible formation_energy: -12.0000 eV' in text
    assert '-100.0000' not in text and '-50.0000' not in text
    assert 'eligible candidates so far: 2' in text
    assert all(len(line) == output.width for line in messages)


def test_resume_reconstructs_totals_without_reprinting_rounds(tmp_path):
    rows = [row(), row(step=2)]
    output, messages = reporter(tmp_path, rows, resumed=True)
    first = [dict(step=0, decisions=[0]), dict(step=1, decisions=[0])]
    offset = journal(tmp_path, first)
    output.progress(1, offset, resumed=True)
    assert 'resumed after committed round 1/2' in '\n'.join(messages)
    assert not any('1/2' in line and 'resumed' not in line for line in messages)
    offset = journal(tmp_path, first + [dict(step=2, decisions=[0])])
    output.progress(2, offset)
    output.finish('COMPLETE')
    text = '\n'.join(messages)
    assert '2 evaluations | 2 accepted | 0 rejected' in text
    assert sum('2/2' in line for line in messages) == 1


@pytest.mark.parametrize('status', ['COMPLETE', 'WAITING', 'EXTINCT', 'FAILED'])
def test_initialization_empty_pool_ascii_and_status(tmp_path, status):
    output, messages = reporter(tmp_path, [row(generation=0, extinct=1)], generation=0, unicode=False)
    output.finish(status)
    output.finish(status)
    text = '\n'.join(messages)
    assert f'{status.lower()} | evaluated: 1/4 | extinct: 1' in text
    assert 'best eligible formation_energy: -' in text
    assert text.isascii()
    assert text.count(status.lower() + ' |') == 1


@pytest.mark.parametrize('level', [logging.INFO, logging.DEBUG])
def test_scoped_logging_preserves_warnings_and_restores_after_failure(level):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(config.formatter)
    old_level = config.logger.level
    config.logger.addHandler(handler)
    config.logger.setLevel(level)
    try:
        with pytest.raises(RuntimeError):
            with bh_logging():
                config.logger.info('routine worker diagnostic')
                config.logger.info('| generation panel |', extra={'gdpx_panel': True})
                config.logger.warning('important warning')
                raise RuntimeError('calculation failed')
        config.logger.info('outside BH')
    finally:
        config.logger.removeHandler(handler)
        config.logger.setLevel(old_level)
    text = stream.getvalue()
    assert '| generation panel |\n' in text
    assert re.search(r'^\d{4}[A-Za-z]{3}\d{2}-\d{2}:\d{2}:\d{2} - INFO: \| generation panel \|$', text, re.MULTILINE)
    assert 'WARNING: important warning' in text
    assert 'INFO: outside BH' in text
    assert ('DEBUG: routine worker diagnostic' in text) == (level == logging.DEBUG)
    assert 'INFO: routine worker diagnostic' not in text


def test_file_formatter_and_ascii_detection(tmp_path):
    from gdpx.exploration.basin_hopping.output import _unicode_supported
    path = tmp_path / 'gdp.out'
    handler = logging.FileHandler(path, encoding='ascii')
    handler.setFormatter(config.formatter)
    config.logger.addHandler(handler)
    try:
        assert not _unicode_supported()
        config.logger.info('+ generation +', extra={'gdpx_panel': True})
    finally:
        config.logger.removeHandler(handler)
        handler.close()
    assert re.fullmatch(r'\d{4}[A-Za-z]{3}\d{2}-\d{2}:\d{2}:\d{2} - INFO: \+ generation \+\n', path.read_text())


@pytest.mark.parametrize('moves', [1000, 100000])
@pytest.mark.parametrize('unicode', [True, False])
def test_round_table_has_inline_best_and_room_for_large_rounds(tmp_path, moves, unicode):
    messages = []
    db = SimpleNamespace(connection=SimpleNamespace(select=lambda **kwargs: iter([row(step=moves)])))
    output = GenerationReporter(db, tmp_path, 1, 1, 2, moves, 4, 'energy',
                                emit=messages.append, unicode=unicode)
    header = next(line for line in messages if 'best energy [eV]' in line)
    offset = journal(tmp_path, [dict(step=0, decisions=[0]), dict(step=moves, decisions=[0])])
    before = len(messages)
    output.progress(moves, offset)
    assert len(messages) == before + 1
    line = messages[-1]
    assert f'{moves}/{moves}' in line and '-10.0000' in line
    assert line.index(f'{moves}/{moves}') == header.index('round')
    assert line.index('-10.0000') + len('-10.0000') == header.index('best energy [eV]') + len('best energy [eV]')
    assert all(len(message) == output.width for message in messages)


@pytest.mark.parametrize('generation,maximum', [(1, 1), (12, 1000), (1000, 1000)])
def test_generation_border_expands_and_contains_move_budget(tmp_path, generation, maximum):
    messages = []
    db = SimpleNamespace(connection=SimpleNamespace(select=lambda **kwargs: iter([])))
    output = GenerationReporter(db, tmp_path, generation, maximum, 2, 1000, 4,
                                'energy', emit=messages.append, resumed=True)
    title = messages[0]
    assert f'basin hopping | generation {generation}/{maximum} | hopping' in title
    assert 'steps/chain: 1000 | resumed' in title
    assert 'chains:' not in title
    assert '...' not in title
    assert not any('chains:' in line or 'objective:' in line for line in messages[1:])
    assert all(len(line) == output.width for line in messages)
