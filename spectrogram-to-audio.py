#!/usr/bin/env python3
"""Spectrogram to Audio - read a picture, write sound.

    ./spectrogram-to-audio.sh                    open the GUI
    ./spectrogram-to-audio.sh image.png          open it with an image loaded
    ./spectrogram-to-audio.sh --cli image.png -o out.mp3    no window

Go through the .sh so the venv is used; venv/bin/python spectrogram-to-audio.py
does the same thing.

The other direction lives in audio-to-spectrogram.py, which shares no code with
this and needs no venv.
"""

from __future__ import annotations

import sys


def main() -> int:
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from spectro.cli import main as cli_main
        return cli_main()

    try:
        from spectro.gui import main as gui_main
    except ImportError as exc:
        print(f"Could not start the GUI: {exc}\n\n"
              "The dependencies live in the project venv. Launch with:\n"
              "    ./spectrogram-to-audio.sh\n"
              "or   venv/bin/python spectrogram-to-audio.py", file=sys.stderr)
        return 1
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
