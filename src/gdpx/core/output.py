"""Shared scrolling boxes and scoped diagnostic logging."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import textwrap

from gdpx import config

_DIAGNOSTICS = ContextVar('output_diagnostics', default=False)


class _DiagnosticsFilter(logging.Filter):
    def filter(self, record):
        if _DIAGNOSTICS.get() and record.levelno == logging.INFO and not getattr(record, 'gdpx_panel', False):
            if not config.logger.isEnabledFor(logging.DEBUG):
                return False
            record.levelno, record.levelname = logging.DEBUG, 'DEBUG'
        return True


@contextmanager
def quiet_logging():
    """Demote only routine GDP messages from this execution context."""
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



_PARENT = ContextVar('output_parent', default=None)


class Box:
    width = 76

    def __init__(self, title, emit=None, unicode=None):
        parent = _PARENT.get()
        self.emit = emit or (parent.line if parent else lambda message: config.logger.info(message, extra={'gdpx_panel': True}))
        self.unicode = (parent.unicode if parent else _unicode_supported()) if unicode is None else unicode
        self.width = parent.width - 4 if parent else max(self.width, len(title) + 6)
        self.border('top', title)

    @contextmanager
    def as_parent(self):
        token = _PARENT.set(self)
        try:
            yield self
        finally:
            _PARENT.reset(token)

    def border(self, kind, title=None):
        glyphs = {'top': ('┌', '┐'), 'middle': ('├', '┤'), 'bottom': ('└', '┘')}
        left, right = glyphs[kind] if self.unicode else ('+', '+')
        rule = '─' if self.unicode else '-'
        content = rule * (self.width - 2)
        if title:
            title = title if len(title) <= self.width - 6 else textwrap.shorten(title, self.width - 6, placeholder='...')
            content = f'{rule} {title} '.ljust(self.width - 2, rule)
        self.emit(left + content + right)

    def line(self, message):
        bar = '│' if self.unicode else '|'
        if not self.unicode:
            message = message.replace('—', '-').replace('Å', 'A')
        for text in textwrap.wrap(message, self.width - 4) or ['']:
            self.emit(f'{bar} {text:<{self.width - 4}} {bar}')


def message(text):
    parent = _PARENT.get()
    if parent:
        parent.line(text)
    else:
        config.logger.info(text, extra={'gdpx_panel': True})
