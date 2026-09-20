"""Versioned JSON/NumPy data and atomic, bounded restart snapshots."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_data(path, data):
    """Publish a JSON document referencing an immutable, non-pickled array file."""
    path = Path(path)
    if path.suffix != '.json':
        raise ValueError('Portable checkpoints require a .json path.')
    arrays = {}
    def encode(value):
        if isinstance(value, (np.ndarray, np.generic)):
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError('Object arrays are not supported in checkpoints.')
            key = f'array_{len(arrays)}'
            arrays[key] = array
            return {'__checkpoint_type__': 'scalar' if isinstance(value, np.generic) else 'array', 'key': key}
        if isinstance(value, dict):
            if all(isinstance(k, str) for k in value) and '__checkpoint_type__' not in value:
                return {k: encode(v) for k, v in value.items()}
            return {'__checkpoint_type__': 'mapping', 'items': [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, tuple):
            return {'__checkpoint_type__': 'tuple', 'items': [encode(v) for v in value]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if isinstance(value, Path):
            return {'__checkpoint_type__': 'path', 'value': str(value)}
        if isinstance(value, bytes):
            return {'__checkpoint_type__': 'bytes', 'value': base64.b64encode(value).decode('ascii')}
        if isinstance(value, float) and not math.isfinite(value):
            return {'__checkpoint_type__': 'float', 'value': str(value)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}.')
    payload = encode(data)
    document = dict(format='gdpx-checkpoint', version=1, data=payload)
    if arrays:
        archive = path.with_name(f'{path.stem}-{uuid.uuid4().hex}.npz')
        with archive.open('wb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        document.update(arrays=archive.name, sha256=_digest(archive))
    write_json(path, document)


def load_data(path):
    path = Path(path)
    if path.suffix != '.json':
        raise ValueError('Legacy pickle checkpoint is not supported; start a new run.')
    with path.open(encoding='utf-8') as stream:
        document = json.load(stream)
    if document.get('format') != 'gdpx-checkpoint' or document.get('version') != 1:
        raise ValueError('Unsupported portable checkpoint format or version.')
    arrays = {}
    if 'arrays' in document:
        name = document['arrays']
        if Path(name).name != name:
            raise ValueError('Invalid checkpoint array filename.')
        archive = path.parent / name
        if _digest(archive) != document['sha256']:
            raise ValueError('Checkpoint array checksum mismatch.')
        with np.load(archive, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
    def decode(value):
        if isinstance(value, list):
            return [decode(v) for v in value]
        if not isinstance(value, dict):
            return value
        kind = value.get('__checkpoint_type__')
        if kind is None:
            return {k: decode(v) for k, v in value.items()}
        if kind in ('array', 'scalar'):
            array = arrays[value['key']]
            return array[()] if kind == 'scalar' else array
        if kind == 'tuple':
            return tuple(decode(v) for v in value['items'])
        if kind == 'mapping':
            return {decode(k): decode(v) for k, v in value['items']}
        if kind == 'path':
            return Path(value['value'])
        if kind == 'bytes':
            return base64.b64decode(value['value'])
        if kind == 'float':
            return float(value['value'])
        raise ValueError(f'Unsupported checkpoint data tag: {kind}.')
    return decode(document['data'])


def publish_snapshot(root, staging, destination, prefix):
    """Publish before pruning; interrupted pruning is repeated on next commit."""
    root, staging, destination = Path(root), Path(staging), Path(destination)
    manifest = root / 'current.json'
    previous = json.loads(manifest.read_text()).get('snapshots', []) if manifest.exists() else []
    if destination.exists():
        shutil.rmtree(destination)  # Only an unpublished/replayed snapshot at this step.
    staging.rename(destination)
    names = [destination.name] + [name for name in previous if name != destination.name]
    write_json(manifest, {'version': 1, 'snapshots': names[:2]})
    for path in root.glob(prefix + '*'):
        if path.is_dir() and path.name not in names[:2]:
            shutil.rmtree(path)


def prune_snapshots(root, prefix):
    root = Path(root)
    names = json.loads((root / 'current.json').read_text())['snapshots']
    for path in root.glob(prefix + '*'):
        if path.is_dir() and path.name not in names:
            shutil.rmtree(path)


def read_snapshot(root, loader):
    """Return the newest validated published snapshot, falling back once."""
    root = Path(root)
    manifest = json.loads((root / 'current.json').read_text())
    if manifest.get('version') != 1:
        raise ValueError('Unsupported checkpoint manifest version.')
    errors = []
    for index, name in enumerate(manifest['snapshots']):
        if Path(name).name != name:
            raise ValueError('Invalid checkpoint directory name.')
        try:
            value = loader(root / name)
            if index:
                write_json(root / 'current.json', dict(version=1, snapshots=manifest['snapshots'][index:]))
            return root / name, value
        except (OSError, ValueError, KeyError, EOFError) as error:
            errors.append(str(error))
    raise ValueError('No complete checkpoint remains: ' + '; '.join(errors))
