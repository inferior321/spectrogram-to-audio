#!/usr/bin/env bash
# Make a spectrogram picture out of audio.
#
# This one needs NO venv and nothing from pip - it is standard library only.
# What it does need is two system packages, so they are checked here rather
# than left to fail with a traceback:
#
#     ffmpeg      does the actual work
#     python3-tk  the GUI toolkit; pip cannot install this one
#
# The other direction is spectrogram-to-audio.sh, which does use the venv.
cd "$(dirname "$0")" || exit 1

PY=python3
command -v "$PY" >/dev/null || { echo "python3 not found." >&2; exit 1; }

# Each entry is appended with its own leading space, so the apt line below
# reads correctly whether one thing is missing or both.
missing=""
"$PY" -c "import tkinter" 2>/dev/null || missing="$missing python3-tk"
command -v ffmpeg >/dev/null || missing="$missing ffmpeg"

if [ -n "$missing" ]; then
    echo "Missing:$missing" >&2
    echo >&2
    echo "  sudo apt install$missing" >&2
    echo >&2
    echo "python3-tk is not on PyPI, so pip and the venv cannot supply it." >&2
    exit 1
fi

command -v ffprobe >/dev/null || \
    echo "note: ffprobe not found - the duration readout and 'split into chunks' will not work." >&2

exec "$PY" audio-to-spectrogram.py "$@"
