"""Streaming helpers for driver-output archives."""

from __future__ import annotations

import contextlib
import os
import pathlib
import tarfile
import tempfile
from collections.abc import Iterable, Iterator
from typing import Optional

import zstandard


ZSTD_ARCHIVE_NAME = "cand.tar.zst"
LEGACY_GZIP_ARCHIVE_NAME = "cand.tgz"


@contextlib.contextmanager
def open_archive(archive_path: pathlib.Path) -> Iterator[tarfile.TarFile]:
    """Open a gzip or Zstandard tar archive for sequential reading."""
    archive_path = pathlib.Path(archive_path)
    if archive_path.name.endswith(".zst"):
        with archive_path.open("rb") as raw:
            decompressor = zstandard.ZstdDecompressor()
            with decompressor.stream_reader(raw, closefd=False) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    yield tar
    else:
        with tarfile.open(archive_path, mode="r:*") as tar:
            yield tar


def find_driver_archive(directory: pathlib.Path) -> Optional[pathlib.Path]:
    """Return the preferred existing driver archive, if any."""
    directory = pathlib.Path(directory)
    for name in (ZSTD_ARCHIVE_NAME, LEGACY_GZIP_ARCHIVE_NAME):
        archive_path = directory / name
        if archive_path.exists():
            return archive_path.absolute()
    return None


def create_zstd_archive(
    archive_path: pathlib.Path,
    entries: Iterable[tuple[pathlib.Path, str]],
    *,
    level: int = 3,
) -> None:
    """Create a Zstandard tar archive and publish it atomically."""
    archive_path = pathlib.Path(archive_path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=archive_path.parent,
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = pathlib.Path(temporary_name)

    try:
        with temporary_path.open("wb") as raw:
            compressor = zstandard.ZstdCompressor(level=level, threads=0, write_checksum=True)
            with compressor.stream_writer(raw, closefd=False) as writer:
                with tarfile.open(fileobj=writer, mode="w|") as tar:
                    for source_path, archive_name in entries:
                        tar.add(source_path, arcname=archive_name)
        os.replace(temporary_path, archive_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
