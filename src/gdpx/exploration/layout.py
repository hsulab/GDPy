"""Shared exploration output layout and persisted restart identity."""

import json
import os
import tempfile
from pathlib import Path


def exploration_directories(count):
    if count < 1:
        raise ValueError('Exploration requires at least one exploration.')
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


def _write_database(path, data):
    """Publish a complete TinyDB file before retiring its old inputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2)
        stream.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def exploration_layout(directory, count=None, *, create=False, scheduler=None):
    """Read layout from scheduler databases; optionally consolidate older files.

    Jobs remain in TinyDB's default table. A separate ``layout`` table stores
    one record, so job queries and mutations cannot mistake it for a job.
    """
    directory = Path(directory)
    reject_legacy_layout(directory)
    metadata = directory / '_meta'
    old_layouts = [path for path in (metadata / 'layout.json', metadata / 'exploration.json')
                   if path.exists()]
    layouts = [json.loads(path.read_text()) for path in old_layouts]
    databases = {}
    providers = set()
    for path in metadata.glob('_*.json'):
        data = json.loads(path.read_text())
        databases[path] = data
        if path.name == '_scheduler.json':
            records = list(data.get('scheduler', {}).values())
            if len(records) != 1 or not isinstance(records[0].get('provider'), str):
                raise ValueError(f'Missing scheduler provider in {path}')
            providers.add(records[0]['provider'])
        else:
            providers.add(path.stem.removesuffix('_jobs').removeprefix('_'))
        if 'layout' in data:
            records = list(data['layout'].values())
            if len(records) != 1:
                raise ValueError(f'Invalid exploration layout: {path}')
            layouts.append(records[0])
    if len(providers) > 1:
        raise ValueError(f'Conflicting scheduler providers in {metadata}: {sorted(providers)}')
    provider = next(iter(providers), scheduler or 'direct')
    if scheduler is not None and scheduler != provider:
        raise ValueError(f'Scheduler provider changed from {provider} to {scheduler}; use a fresh directory.')
    layout = layouts[0] if layouts else None
    if layout is not None:
        saved_count = layout['count']
        if (layout.get('version') != 1 or layout.get('directories') != exploration_directories(saved_count)
                or any(other != layout for other in layouts)):
            raise ValueError(f'Invalid or conflicting exploration layouts in {metadata}')
        if count is not None and count != saved_count:
            raise ValueError(f'Exploration count changed from {saved_count} to {count}; use a fresh directory.')
    elif databases:
        raise ValueError(f'Missing exploration layout in {metadata}; cannot safely resume job records.')
    elif count is not None:
        layout = dict(version=1, count=count, directories=exploration_directories(count))
    else:
        return None

    if create:
        # Validate all inputs before publishing one database and retiring sources.
        merged = None
        for data in databases.values():
            candidate = dict(data, layout={'1': layout}, scheduler={'1': {'provider': provider}})
            if merged is not None and merged != candidate:
                raise ValueError(f'Conflicting scheduler records in {metadata}')
            merged = candidate
        if merged is None:
            merged = {'_default': {}, 'layout': {'1': layout},
                      'scheduler': {'1': {'provider': provider}}}
        target = metadata / '_scheduler.json'
        if databases.get(target) != merged:
            _write_database(target, merged)
        for path in list(databases) + old_layouts:
            if path != target:
                path.unlink()
    return layout['directories']
