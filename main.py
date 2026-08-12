#!/usr/bin/env python3
"""Spectrogram to Audio - entry point.

    ./run.sh                        launch the GUI
    ./run.sh image.png              launch the GUI with an image already open
    ./run.sh --cli image.png -o out.mp3     convert without opening a window

Run it through run.sh (or venv/bin/python main.py) so the venv is used.
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
              "    ./run.sh\n"
              "or   venv/bin/python main.py", file=sys.stderr)
        return 1
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
