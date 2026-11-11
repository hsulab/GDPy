#!/usr/bin/env bash

set -euo pipefail

docs_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
port="${1:-8000}"

if ! [[ "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
    echo "Usage: $0 [port]" >&2
    echo "port must be an integer from 1 to 65535" >&2
    exit 2
fi

if ! command -v python >/dev/null 2>&1; then
    echo "Activate a Python environment before running the documentation server." >&2
    exit 1
fi

if ! python -c "import myst_parser, sphinx, sphinx_autobuild, sphinx_copybutton, sphinx_rtd_theme" >/dev/null 2>&1; then
    echo "The active Python environment is missing the documentation dependencies." >&2
    echo "Install them once with:" >&2
    echo "  python -m pip install -r \"$docs_dir/requirements.txt\"" >&2
    exit 1
fi

python -m sphinx -M clean "$docs_dir/source" "$docs_dir/build"

echo "Serving gdpx documentation with live reload at http://127.0.0.1:$port"
exec python -m sphinx_autobuild \
    --host 127.0.0.1 \
    --port "$port" \
    -a \
    -W --keep-going \
    "$docs_dir/source" "$docs_dir/build/html"
