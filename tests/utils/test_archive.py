import pathlib
import tarfile

import pytest

import gdpx.utils.archive as archive_module
from gdpx.utils.archive import (
    LEGACY_GZIP_ARCHIVE_NAME,
    ZSTD_ARCHIVE_NAME,
    create_zstd_archive,
    find_driver_archive,
    open_archive,
)


def _archive_contents(archive_path: pathlib.Path) -> dict[str, bytes]:
    contents = {}
    with open_archive(archive_path) as tar:
        for member in tar:
            if member.isfile():
                extracted = tar.extractfile(member)
                assert extracted is not None
                contents[member.name] = extracted.read()
    return contents


def test_zstd_archive_round_trip(tmp_path):
    source = tmp_path / "cand0"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "trajectory.xyz").write_text("2\nframe\nH 0 0 0\nH 0 0 1\n")
    (nested / "restart.bin").write_bytes(bytes(range(256)))
    archive_path = tmp_path / ZSTD_ARCHIVE_NAME

    create_zstd_archive(archive_path, [(source, source.name)])

    assert archive_path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
    assert _archive_contents(archive_path) == {
        "cand0/nested/restart.bin": bytes(range(256)),
        "cand0/trajectory.xyz": b"2\nframe\nH 0 0 0\nH 0 0 1\n",
    }


def test_open_archive_reads_legacy_gzip(tmp_path):
    source = tmp_path / "cand0"
    source.mkdir()
    (source / "result.txt").write_text("legacy")
    archive_path = tmp_path / LEGACY_GZIP_ARCHIVE_NAME
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source, arcname=source.name)

    assert _archive_contents(archive_path)["cand0/result.txt"] == b"legacy"


def test_find_driver_archive_prefers_zstd(tmp_path):
    gzip_path = tmp_path / LEGACY_GZIP_ARCHIVE_NAME
    zstd_path = tmp_path / ZSTD_ARCHIVE_NAME
    gzip_path.touch()
    assert find_driver_archive(tmp_path) == gzip_path.absolute()

    zstd_path.touch()
    assert find_driver_archive(tmp_path) == zstd_path.absolute()


def test_failed_archive_keeps_sources_and_removes_temporary_file(tmp_path):
    source = tmp_path / "cand0"
    source.mkdir()
    (source / "result.txt").write_text("keep me")
    archive_path = tmp_path / ZSTD_ARCHIVE_NAME

    with pytest.raises(FileNotFoundError):
        create_zstd_archive(
            archive_path,
            [(source, source.name), (tmp_path / "missing", "missing")],
        )

    assert source.is_dir()
    assert not archive_path.exists()
    assert not list(tmp_path.glob(f".{ZSTD_ARCHIVE_NAME}.*.tmp"))


def test_failed_publish_keeps_sources(monkeypatch, tmp_path):
    source = tmp_path / "cand0"
    source.mkdir()
    (source / "result.txt").write_text("keep me")
    archive_path = tmp_path / ZSTD_ARCHIVE_NAME

    def fail_replace(source_path, destination_path):
        raise OSError("injected publication failure")

    monkeypatch.setattr(archive_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected publication failure"):
        create_zstd_archive(archive_path, [(source, source.name)])

    assert source.is_dir()
    assert not archive_path.exists()
    assert not list(tmp_path.glob(f".{ZSTD_ARCHIVE_NAME}.*.tmp"))
