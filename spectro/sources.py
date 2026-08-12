"""Where an image came from, as an explicit choice.

Nothing here guesses.  You say which tool drew the picture and which colour
scheme it was set to, and that fixes how pixels are read.  The frequency axis
is chosen separately, because every source below can draw on any axis, and a
combined list would be six times longer for no gain.

Every gradient here is the real one.  There is no "close enough" tier: an
entry named after a tool was either derived from that tool's own exports
(Audacity) or read out of the colour bar the tool itself draws (ffmpeg, SoX,
Spek), and matplotlib's were lifted from the library.

Entries that merely borrowed a similar-looking gradient used to live here -
Sonic Visualiser, Adobe Audition, iZotope RX - and were removed.  They added no
capability, because each was an alias for a matplotlib map already in the list;
all they added was a tool's name implying a verification that had not happened.
If you have an image from a tool that is not listed, pick the closest-looking
gradient, or produce two exports of the same audio from it - one monochrome,
one in colour - and its table can be derived exactly.

Each entry states what it is:

    exact        the table is the real one
    generic      not a specific tool - a plain reading of brightness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import colormap


@dataclass(frozen=True)
class Source:
    """One selectable way of reading an image."""

    label: str                       # what the dropdown shows
    scheme: str                      # a name from colormap.available()
    accuracy: str                    # exact | generic
    note: str = ""                   # shown under the dropdown
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def is_separator(self) -> bool:
        return self.scheme == ""


def _sep(label: str) -> Source:
    return Source(label=label, scheme="", accuracy="")


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------

AUDACITY = [
    Source("Audacity — Color (default)", "Audacity Color (default)", "exact",
           "Black → purple → orange → pale yellow. Table derived from real "
           "exports; it fits them to 1.9/255."),
    Source("Audacity — Color (classic)", "Audacity Color (classic)", "exact",
           "Pale blue-white → blue → magenta → red → pale pink-white. Exact, "
           "but note both ends are near-white and only 17/255 apart: fine from "
           "a PNG, unreliable from a JPEG or a rescaled screenshot."),
    Source("Audacity — Grayscale", "Audacity Grayscale", "exact",
           "White is silence, black is energy. This is the scheme this project "
           "was built around."),
    Source("Audacity — Inverse Grayscale", "Audacity Inverse Grayscale", "exact",
           "Black is silence, white is energy."),
]

# Exact tables, and the ones most Python-generated spectrograms use.
#
# FFmpeg has modes with several of these names, and they are NOT the same
# tables - measured, its magma is 13/255 away from matplotlib's and its terrain
# 72/255, because it re-anchors each one to run black to white.  FFmpeg images
# belong under the ffmpeg heading, which carries its real palettes.
MATPLOTLIB_MAIN = ["viridis", "magma", "inferno", "plasma", "cividis", "turbo",
                   "jet", "gray", "hot", "bone", "copper", "cool"]
MATPLOTLIB_REST = ["afmhot", "gnuplot2", "nipy_spectral", "rainbow", "CMRmap",
                   "ocean", "terrain", "twilight", "cubehelix", "YlGnBu",
                   "Spectral", "coolwarm"]

# ffmpeg's showspectrumpic modes, in the order its own -h filter lists them.
# Every one is exact: the tables were read out of the colour bar ffmpeg draws
# in its legend, by tools/extract_tool_palettes.py.
FFMPEG_MODES = ["intensity", "rainbow", "moreland", "nebulae", "fire", "fiery",
                "fruit", "cool", "magma", "green", "viridis", "plasma",
                "cividis", "terrain", "channel"]

# SoX's spectrogram flags.  Its display is calibrated by documented defaults -
# 120 dB of range topping out at 0 dBFS - so the level mapping can be set
# correctly too, which is not possible for most sources.
SOX_LEVELS = {"level_mapping": "Plain dB range", "range_db": 120.0,
              "min_freq": 0.0}
SOX_MODES = [
    ("default", "sox: default", "Black → purple → red → orange → white."),
    ("light background (-l)", "sox: light", "White page, dark energy."),
    ("monochrome (-m)", "sox: mono", "Black → white grey ramp."),
    ("monochrome light (-m -l)", "sox: mono light", "White page, dark energy."),
    ("high colour (-h)", "sox: high colour", "Wider, more saturated ramp."),
]

# Spek states its own calibration on the legend it draws: 0 dB at the top of
# the bar, -120 dB at the bottom.
SPEK_LEVELS = {"level_mapping": "Plain dB range", "range_db": 120.0,
               "min_freq": 0.0}

GENERIC = [
    Source("Greyscale — light background is quiet", "Greyscale (light = quiet)",
           "generic", "Any mono image where white means silence."),
    Source("Greyscale — dark background is quiet", "Greyscale (dark = quiet)",
           "generic", "Any mono image where black means silence."),
    Source("Any picture → sound (no dB curve)", "Greyscale (dark = quiet)",
           "generic",
           "Treats brightness as amplitude directly instead of undoing a dB "
           "display curve. This is the setting for sonifying artwork rather "
           "than reading back a real spectrogram; Gain and Range are ignored.",
           {"level_mapping": "Linear amplitude", "min_freq": 20.0}),
]


def catalogue() -> list[Source]:
    """The full list, with separators, in the order the dropdown shows it."""
    items: list[Source] = [_sep("── Audacity ──")]
    items += AUDACITY

    items.append(_sep("── Python: matplotlib / librosa (exact) ──"))
    for name in MATPLOTLIB_MAIN + MATPLOTLIB_REST:
        key = f"matplotlib: {name}"
        if colormap.get_lut(key) is not None:
            items.append(Source(f"matplotlib — {name}", key, "exact"))

    items.append(_sep("── ffmpeg showspectrumpic (exact) ──"))
    for mode in FFMPEG_MODES:
        key = f"ffmpeg: {mode}"
        if colormap.get_lut(key) is not None:
            items.append(Source(f"ffmpeg — {mode}", key, "exact"))

    items.append(_sep("── SoX spectrogram (exact) ──"))
    for label, key, note in SOX_MODES:
        if colormap.get_lut(key) is not None:
            items.append(Source(f"SoX — {label}", key, "exact",
                                note + " SoX's documented defaults are 120 dB "
                                "of range topping out at 0 dBFS, so the level "
                                "mapping is set for you as well.",
                                dict(SOX_LEVELS)))

    if colormap.get_lut("spek: default") is not None:
        items.append(_sep("── Spek (exact) ──"))
        items.append(Source(
            "Spek", "spek: default", "exact",
            "Black → purple → red → orange → cream. Read out of Spek's own "
            "legend bar, which is labelled 0 dB at the top and -120 dB at the "
            "bottom, so the level mapping is set for you as well. Note that "
            "Spek's 0 dB reference is not SoX's: on the same recording it "
            "reads about 13 dB lower, which is the two tools normalising "
            "differently rather than either being wrong.",
            dict(SPEK_LEVELS)))

    items.append(_sep("── Generic ──"))
    items += GENERIC

    return items


def find(label: str) -> Source | None:
    for s in catalogue():
        if s.label == label and not s.is_separator:
            return s
    return None


def label_for_scheme(scheme: str) -> str | None:
    """The catalogue entry that a bare scheme name corresponds to."""
    for s in catalogue():
        if not s.is_separator and s.scheme == scheme and not s.overrides:
            return s.label
    return None
