#!/usr/bin/env bash

set -euo pipefail

docs_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
port="${1:-8000}"

if ! [[ "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
    echo "Usage: $0 [port]" >&2
    echo "port must be an integer from 1 to 65535" >&2
    exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to run the documentation server" >&2
    exit 1
fi

if ! conda run -n catorch3 python -c "import myst_parser, sphinx, sphinx_autobuild" >/dev/null 2>&1; then
    echo "The catorch3 environment is missing the documentation dependencies." >&2
    echo "Install them once with:" >&2
    echo "  conda run -n catorch3 python -m pip install -r $docs_dir/requirements.txt" >&2
    exit 1
fi

conda run --no-capture-output -n catorch3 \
    make -C "$docs_dir" clean

echo "Serving GDPy documentation with live reload at http://127.0.0.1:$port"
exec conda run --no-capture-output -n catorch3 \
    sphinx-autobuild \
    --host 127.0.0.1 \
    --port "$port" \
    -W --keep-going \
    "$docs_dir/source" "$docs_dir/build/html"
