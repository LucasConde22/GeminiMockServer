#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

PORT="${PORT:-5050}"
HOST="${HOST:-0.0.0.0}"

echo "============================================================"
echo "Starting Gemini Mock Server with Antigravity on $HOST:$PORT"
echo "============================================================"

# Try running with Flask server.py; if Flask is not installed, fallback to standalone_server.py
if python3 -c "import flask" >/dev/null 2>&1; then
    exec python3 server.py --host "$HOST" --port "$PORT" "$@"
else
    echo "[NOTICE] Flask not found, falling back to standalone_server.py (zero dependencies)..."
    exec python3 standalone_server.py --host "$HOST" --port "$PORT" "$@"
fi
