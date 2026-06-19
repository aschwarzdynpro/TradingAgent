#!/bin/bash
# SessionStart hook: prepare a Python venv so tests/backtest run in web sessions.
# Idempotent and non-interactive. Synchronous (blocks session start until ready).
set -euo pipefail

cd "$CLAUDE_PROJECT_DIR"

VENV="$CLAUDE_PROJECT_DIR/.venv"
PY="${PYTHON_BIN:-python3}"

# 1) Create the virtualenv once; reused (and cached) on subsequent runs.
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi

# 2) Install the project + dev extras (pytest). `install -e` benefits from the
#    cached container state on later runs.
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -e ".[dev]"

# 3) Persist the venv for the rest of the session so `python`/`pytest` resolve
#    to it without manual activation.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export VIRTUAL_ENV=\"$VENV\""
    echo "export PATH=\"$VENV/bin:\$PATH\""
  } >> "$CLAUDE_ENV_FILE"
fi

echo "session-start: venv ready, dependencies installed."
