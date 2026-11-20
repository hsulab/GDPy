"""Display the Git-derived version recorded at build/install time."""

from importlib import metadata


def version_line():
    try:
        version = metadata.version("gdpx")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return f"version: {version}"
