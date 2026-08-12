#!/usr/bin/env bash
# Launch the app using the project venv, whatever directory you call it from.
cd "$(dirname "$0")" || exit 1
if [ ! -x venv/bin/python ]; then
    echo "venv missing - recreate it with:" >&2
    echo "    python3 -m venv --copies venv && venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
exec venv/bin/python main.py "$@"
