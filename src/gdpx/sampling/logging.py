"""Scoped operator diagnostics; independent of structures and sampling decisions."""
from contextlib import contextmanager
import json
import logging
from pathlib import Path

from gdpx import config


class MoveLog:
    """Append human-readable diagnostics without changing the global log level."""
    def __init__(self, path, generation, operators, probabilities):
        self.path = Path(path)
        self.generation = generation
        self.operators = operators
        self.probabilities = probabilities
        self.logger = logging.Logger(f'gdpx.moves.{generation}', level=logging.DEBUG)
        self.logger.propagate = False
        self.handler = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handler = logging.FileHandler(self.path, mode='a', encoding='utf-8')
        self.handler.setFormatter(config.formatter)
        self.logger.addHandler(self.handler)
        try:
            self.write('invocation begins; diagnostic history only; events.jsonl defines committed outcomes')
            for index, (operator, probability) in enumerate(zip(self.operators, self.probabilities)):
                self.write('configuration: ' + json.dumps(operator.as_dict(), sort_keys=True, default=str),
                           operator=f'{index}:{operator.name}')
                self.write(f'selection probability: {probability:.6g}', operator=f'{index}:{operator.name}')
        except BaseException:
            self.logger.removeHandler(self.handler)
            self.handler.close()
            raise
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.write(f'invocation failed: {kind.__name__}: {error}' if error else 'invocation ends',
                       level=logging.ERROR if error else logging.INFO)
        finally:
            self.logger.removeHandler(self.handler)
            self.handler.close()

    def write(self, text, *, level=logging.INFO, **context):
        if level == logging.DEBUG and not config.logger.isEnabledFor(logging.DEBUG):
            return
        fields = dict(generation=self.generation, round='-', chain='-', segment='-', parent='-', operator='-')
        fields.update(context)
        prefix = ' '.join(f'{key}={value}' for key, value in fields.items())
        for line in str(text).splitlines() or ['']:
            self.logger.log(level, f'{prefix} | {line}')

    @contextmanager
    def operator(self, move, **context):
        previous_print, previous_debug = move._print, move._debug
        move._print = lambda message: self.write(message, **context)
        move._debug = lambda message: self.write(message, level=logging.DEBUG, **context)
        try:
            yield
        except Exception as error:
            self.write(f'proposal failed: {type(error).__name__}: {error}', level=logging.ERROR, **context)
            raise
        finally:
            move._print, move._debug = previous_print, previous_debug
