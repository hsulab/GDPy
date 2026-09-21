"""Scrolling BH progress blocks, independent of structure ownership and RNGs."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import time
import textwrap

from gdpx import config


_DIAGNOSTICS = ContextVar('bh_diagnostics', default=False)


class _DiagnosticsFilter(logging.Filter):
    def filter(self, record):
        if _DIAGNOSTICS.get() and record.levelno == logging.INFO and not getattr(record, 'gdpx_panel', False):
            if not config.logger.isEnabledFor(logging.DEBUG):
                return False
            record.levelno, record.levelname = logging.DEBUG, 'DEBUG'
        return True


@contextmanager
def bh_logging():
    """Demote only routine GDP messages from this BH invocation."""
    token = _DIAGNOSTICS.set(True)
    filter_ = _DiagnosticsFilter()
    config.logger.addFilter(filter_)
    try:
        yield
    finally:
        config.logger.removeFilter(filter_)
        _DIAGNOSTICS.reset(token)


def _unicode_supported():
    for handler in config.logger.handlers:
        encoding = getattr(getattr(handler, 'stream', None), 'encoding', None)
        if encoding:
            try:
                '┌─│├└┐┤┘—'.encode(encoding)
            except (UnicodeError, LookupError):
                return False
    return True


class GenerationReporter:
    width = 76

    def __init__(self, database, directory, generation, maximum_generation, chains,
                 moves, initial_size, objective, resumed=False, emit=None, unicode=None):
        self.database = database
        self.directory = directory
        self.generation, self.moves = generation, moves
        self.initial_size = initial_size
        self.objective = objective
        self.started = time.monotonic()
        self.emit = emit or (lambda message: config.logger.info(message, extra={'gdpx_panel': True}))
        self.unicode = _unicode_supported() if unicode is None else unicode
        self.step, self.offset = 0, 0
        self.invalid, self.restarts = 0, 0
        self.rows, self.current = [], []
        self.closed = False
        headers = ['round', 'eval', 'accept', 'reject', 'invalid', 'extinct', 'restart',
                   f'best {objective} [eV]']
        count_width = len(str(chains))
        self.column_widths = [max(len(headers[0]), 2 * len(str(moves)) + 1)]
        self.column_widths += [max(len(label), count_width) for label in headers[1:-1]]
        self.column_widths.append(max(len(headers[-1]), 14))
        if generation:
            self.width = max(self.width, sum(self.column_widths) + 2 * (len(headers) - 1) + 4)
        title = f'basin hopping | generation {generation}/{maximum_generation} | '
        title += 'initialization' if generation == 0 else 'hopping'
        if generation:
            title += f' | moves/chain: {moves}'
        if resumed:
            title += ' | resumed'
        self.width = max(self.width, len(title) + 6)
        self.border('top', title)
        if generation:
            self.table_row(headers)
        else:
            self.line(f'initial candidates: {initial_size}   chains: {chains}')
            self.line(f'objective: {objective} [eV]')

    def border(self, kind, title=None):
        glyphs = {'top': ('┌', '┐'), 'middle': ('├', '┤'), 'bottom': ('└', '┘')}
        left, right = glyphs[kind] if self.unicode else ('+', '+')
        rule = '─' if self.unicode else '-'
        content = rule * (self.width - 2)
        if title:
            content = f'{rule} {title} '.ljust(self.width - 2, rule)
        self.emit(left + content + right)

    def line(self, message):
        bar = '│' if self.unicode else '|'
        if not self.unicode:
            message = message.replace('—', '-')
        for text in textwrap.wrap(message, self.width - 4) or ['']:
            self.emit(f'{bar} {text:<{self.width - 4}} {bar}')

    def table_row(self, values):
        cells = [f'{str(value):<{width}}' if index == 0 else f'{str(value):>{width}}'
                 for index, (value, width) in enumerate(zip(values, self.column_widths))]
        self.line('  '.join(cells))

    def refresh(self):
        # ASE rows expose scores and provenance without constructing/copying Atoms.
        self.rows = [row for row in self.database.connection.select(
                     relaxed=1, columns=['id', 'key_value_pairs', 'data'])
                     if row.get('generation', 0) < self.generation or
                     (row.get('generation', 0) == self.generation and
                      (not self.generation or row.data.get('round', 0) <= self.step))]
        self.current = [row for row in self.rows if row.get('generation', 0) == self.generation]

    def best(self):
        scores = [row.get('target') for row in self.rows
                  if not row.get('extinct', 0) and row.get('target') is not None]
        return f'{min(scores):.4f} eV' if scores else '—'

    def progress(self, step, offset, resumed=False):
        """Observe a published journal prefix; never report uncommitted results."""
        self.step = step
        last = None
        with (self.directory / 'events.jsonl').open('rb') as stream:
            stream.seek(self.offset)
            while stream.tell() < offset:
                event = json.loads(stream.readline())
                if event['step']:
                    self.invalid += event['decisions'].count(2)
                    self.restarts += event['decisions'].count(4)
                last = event
        self.offset = offset
        self.refresh()
        if resumed:
            if step:
                self.line(f'resumed after committed round {step}/{self.moves}')
            return
        if not step:
            return
        rows = [row for row in self.current if row.data.get('round') == step]
        accepted = sum(bool(row.data.get('accepted')) for row in rows)
        extinct = sum(bool(row.get('extinct', 0)) for row in rows)
        decisions = last['decisions']
        best = self.best().removesuffix(' eV')
        if len(best) > self.column_widths[-1]:
            best = f'{float(best):.6e}'
        self.table_row([f'{step}/{self.moves}', len(rows), accepted, len(rows)-accepted,
                        decisions.count(2), extinct, decisions.count(4), best])

    def finish(self, status, detail=None):
        if self.closed:
            return
        status = status.lower()
        self.refresh()
        self.border('middle')
        if self.generation:
            accepted = sum(bool(row.data.get('accepted')) for row in self.current)
            self.line(f'{status} | {len(self.current)} evaluations | {accepted} accepted | '
                      f'{len(self.current)-accepted} rejected')
            extinct = sum(bool(row.get('extinct', 0)) for row in self.current)
            self.line(f'{self.invalid} invalid | {extinct} extinct | {self.restarts} restarts')
        else:
            extinct = sum(bool(row.get('extinct', 0)) for row in self.current)
            self.line(f'{status} | evaluated: {len(self.current)}/{self.initial_size} | extinct: {extinct}')
        self.line(f'eligible candidates so far: {sum(not row.get("extinct", 0) for row in self.rows)}')
        self.line(f'best eligible {self.objective}: {self.best()}')
        if detail:
            self.line(detail)
        self.line(f'elapsed this invocation: {time.monotonic() - self.started:.1f} s')
        self.border('bottom')
        self.closed = True
