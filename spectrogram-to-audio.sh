#!/usr/bin/env bash
# Read a spectrogram picture and write audio.  Uses the project venv, whatever
# directory you call this from.
#
# The other direction is audio-to-spectrogram.sh, which needs no venv at all.
cd "$(dirname "$0")" || exit 1

if [ ! -x venv/bin/python ]; then
    echo "venv missing - create it with:" >&2
    echo "    python3 -m venv --copies venv" >&2
    echo "    venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

exec venv/bin/python spectrogram-to-audio.py "$@"
