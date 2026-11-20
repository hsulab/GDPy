import pytest

from gdpx import version


@pytest.mark.parametrize("installed_version", [
    "0.1.0", "0.1.1.dev252+gc35f4f84", "0.1.1.dev252+gc35f4f84.d20261120",
])
def test_version_line(monkeypatch, installed_version):
    def package_version(name):
        assert name == "gdpx"
        return installed_version

    monkeypatch.setattr(version.metadata, "version", package_version)
    assert version.version_line() == f"version: {installed_version}"


def test_uninstalled_version(monkeypatch):
    def package_version(name):
        raise version.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(version.metadata, "version", package_version)
    assert version.version_line() == "version: unknown"
