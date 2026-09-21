"""Shared exploration output layout and persisted restart identity."""

import json
from pathlib import Path


def expedition_directories(count):
    if count < 1:
        raise ValueError('Exploration requires at least one expedition.')
    if count == 1:
        return ['.']
    width = 2 * ((len(str(count)) + 1) // 2)
    return [f'expo.{index:0{width}d}' for index in range(count)]


def reject_legacy_layout(directory):
    directory = Path(directory)
    legacy = list(directory.glob('_*_jobs.json'))
    legacy.extend(path for path in directory.glob('expedition-*') if path.is_dir())
    if legacy:
        raise RuntimeError(
            f'Legacy exploration layout in {directory}: {legacy[0].name}. '
            'Use a fresh working directory; automatic migration and legacy resume are not supported.'
        )


def exploration_layout(directory, count=None, *, create=False):
    """Resolve saved directories, rejecting incompatible restarts before writes."""
    directory = Path(directory)
    reject_legacy_layout(directory)
    manifest = directory / '_meta' / 'exploration.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        saved_count = data['count']
        if data.get('version') != 1 or data.get('directories') != expedition_directories(saved_count):
            raise ValueError(f'Invalid exploration layout: {manifest}')
        if count is not None and count != saved_count:
            raise ValueError(f'Exploration count changed from {saved_count} to {count}; use a fresh directory.')
        return data['directories']
    if count is None:
        return None
    directories = expedition_directories(count)
    if create:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(dict(version=1, count=count, directories=directories), indent=2) + '\n')
    return directories
