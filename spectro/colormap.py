"""Colour schemes: turning coloured spectrogram pixels back into levels.

A colour spectrogram is drawn by feeding a level 0..1 through a 256-entry
gradient.  To read one back we need that gradient, and then for each pixel the
index whose colour is nearest.

Three sources of gradients are bundled, plus greyscale:

* `colormaps.npz` - exact tables lifted from matplotlib (viridis, magma, jet,
  turbo and friends).  Most tools in the wild use one of these or something
  very close to them.
* `audacity.npz` - tables derived from real Audacity exports, so its own
  schemes are exact rather than guessed.
* `tools.npz` - ffmpeg's and SoX's own palettes, read straight out of the
  colour bar each draws in its legend.  Note that ffmpeg's modes are NOT the
  matplotlib maps they are named after: its magma differs by 13/255 and its
  terrain by 72/255, because it re-anchors each to run black to white.
* Greyscale and inverse greyscale, which are computed rather than stored.

Which gradient an image used is your choice, not a guess: see `sources.py` for
the catalogue the interface offers.  `learn_lut_from_pair` is not part of that
path - it is how `tools/derive_audacity_luts.py` builds the Audacity tables in
the first place, out of reference exports.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
LUM = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Schemes that are computed, not stored.
BUILTIN = ["Greyscale (light = quiet)", "Greyscale (dark = quiet)"]

# Shown first in the GUI because they are the ones this project was built for.
PREFERRED_ORDER = [
    "Audacity Grayscale",
    "Audacity Inverse Grayscale",
    "Audacity Color (default)",
    "Audacity Color (classic)",
]


@lru_cache(maxsize=1)
def _load_bundled() -> dict[str, np.ndarray]:
    """All stored gradients, as name -> (256, 3) uint8."""
    out: dict[str, np.ndarray] = {}
    for filename, prefix in (("audacity.npz", ""), ("colormaps.npz", "matplotlib: "),
                             ("tools.npz", "")):
        path = DATA_DIR / filename
        if not path.exists():
            continue
        try:
            with np.load(path) as data:
                for key in data.files:
                    out[f"{prefix}{key}"] = np.ascontiguousarray(data[key])
        except (OSError, ValueError):
            continue      # a damaged data file must not stop the app starting
    return out


def available() -> list[str]:
    """Every selectable scheme name, best-known first."""
    bundled = _load_bundled()
    names = [n for n in PREFERRED_ORDER if n in bundled]
    names += sorted(n for n in bundled if n not in names)
    return BUILTIN + names


def get_lut(name: str) -> np.ndarray | None:
    """The (256, 3) uint8 table for `name`, or None for computed schemes."""
    return _load_bundled().get(name)


# --------------------------------------------------------------------------
# Reading levels out of an image
# --------------------------------------------------------------------------

def _unique_colors(rgb8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct colours, an index image, and how often each colour occurs."""
    flat = rgb8.reshape(-1, 3)
    # Pack into one integer so np.unique runs on a 1-D array: far faster than
    # unique-by-row, which matters because these images are megapixel sized.
    packed = (flat[:, 0].astype(np.uint32) << 16 |
              flat[:, 1].astype(np.uint32) << 8 |
              flat[:, 2].astype(np.uint32))
    vals, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    colors = np.stack([(vals >> 16) & 255, (vals >> 8) & 255, vals & 255], axis=1)
    return colors.astype(np.float32), inverse.reshape(rgb8.shape[:2]), counts


def level_from_lut(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map each pixel to the level whose gradient colour is nearest.

    Works on the distinct colours only - a spectrogram drawn from a 256-entry
    gradient has a few hundred of them at most, so this is a tiny search even
    for a megapixel image.
    """
    rgb8 = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    colors, inverse, _ = _unique_colors(rgb8)
    table = lut.astype(np.float32)

    # (n_colors, 256) distance matrix, chunked so memory stays bounded.
    idx = np.empty(len(colors), dtype=np.int32)
    step = 4096
    for a in range(0, len(colors), step):
        chunk = colors[a:a + step]
        d = ((chunk[:, None, :] - table[None, :, :]) ** 2).sum(axis=2)
        idx[a:a + step] = np.argmin(d, axis=1)

    return (idx[inverse] / 255.0).astype(np.float32)


def level_from_scheme(rgb: np.ndarray, scheme: str) -> np.ndarray:
    """Level image for any scheme name, computed or bundled.

    Always returns 1.0 = loud, whatever the scheme's own polarity.
    """
    if scheme == "Greyscale (light = quiet)":
        return np.clip(1.0 - rgb @ LUM, 0.0, 1.0)
    if scheme == "Greyscale (dark = quiet)":
        return np.clip(rgb @ LUM, 0.0, 1.0)
    lut = get_lut(scheme)
    if lut is None:
        return np.clip(rgb @ LUM, 0.0, 1.0)
    return level_from_lut(rgb, lut)


# --------------------------------------------------------------------------
# Recognising which gradient an image used
# --------------------------------------------------------------------------

def fit_error(rgb: np.ndarray, scheme: str, max_colors: int = 4000) -> float:
    """Mean distance, 0..441, from the image's colours to a gradient.

    A measurement, not a guess: nothing calls this to decide anything, because
    the colour scheme is chosen explicitly.  It exists so the derived tables can
    be checked against real exports - a table that fits its own export to about
    1/255 while the wrong table scores 187/255 is a table worth trusting.
    """
    rgb8 = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    colors, _, counts = _unique_colors(rgb8)
    if len(colors) > max_colors:
        keep = np.argsort(counts)[::-1][:max_colors]
        colors, counts = colors[keep], counts[keep]
    weight = counts / counts.sum()

    lut = get_lut(scheme)
    if lut is None:                       # computed greyscale schemes
        spread = np.abs(colors.max(axis=1) - colors.min(axis=1))
        return float((spread * weight).sum())

    table = lut.astype(np.float32)
    d = np.sqrt(((colors[:, None, :] - table[None, :, :]) ** 2).sum(axis=2))
    return float((d.min(axis=1) * weight).sum())


# --------------------------------------------------------------------------
# Learning a gradient from a matched pair of images
# --------------------------------------------------------------------------

def _drop_off_curve(colors: np.ndarray, counts: np.ndarray,
                    factor: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Remove colours that are not part of the gradient.

    Borders, gridlines, cursors and selection tints are the problem, and they
    cannot be found by rarity alone: a window border easily outnumbers the
    brightest entries of the gradient, so filtering on pixel counts throws away
    the top of the table and keeps the chrome.  Geometry separates them
    cleanly instead - gradient colours have an immediate neighbour a couple of
    units away in RGB, while chrome sits on its own.
    """
    if len(colors) < 8:
        return colors, counts
    d = ((colors[:, None, :] - colors[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d, np.inf)
    nearest = np.sqrt(d.min(axis=1))
    keep = nearest <= max(4.0, factor * float(np.median(nearest)))
    if keep.sum() < 8:
        return colors, counts
    return colors[keep], counts[keep]


def learn_lut_from_pair(color_rgb: np.ndarray, gray_rgb: np.ndarray,
                        gray_light_is_quiet: bool = True) -> np.ndarray:
    """Derive a gradient from two exports of the same audio at the same axis.

    This is the accurate route, because it needs no guess about which order the
    colours go in: the greyscale export says outright how loud each point was,
    so every colour can be ranked by the loudness it stood for.  Each colour's
    rank is taken as the MEDIAN greyscale level over all the pixels where that
    colour appears, which shrugs off the fact that two exports are never
    aligned to the pixel.

    The two images must show the same audio on the same frequency axis - a
    linear export and a mel export of the same sound are different pictures and
    pairing them produces nonsense.  They need not be the same size, and they
    need not share Gain and Range: only the ORDER of the levels is used, and
    that is the same whatever the exposure.
    """
    ref = (1.0 - gray_rgb @ LUM) if gray_light_is_quiet else (gray_rgb @ LUM)

    # Put both on one grid so pixels correspond.  Coarse enough to absorb the
    # few-pixel size difference between exports, fine enough that a colour's
    # sample of the reference stays large.
    from PIL import Image
    h = min(color_rgb.shape[0], ref.shape[0])
    w = min(color_rgb.shape[1], ref.shape[1])
    ref_img = Image.fromarray((np.clip(ref, 0, 1) * 255).astype(np.uint8))
    ref_fit = np.asarray(ref_img.resize((w, h), Image.BILINEAR),
                         dtype=np.float32).ravel() / 255.0
    col_img = Image.fromarray(
        np.rint(np.clip(color_rgb, 0, 1) * 255).astype(np.uint8))
    col_fit = np.asarray(col_img.resize((w, h), Image.NEAREST), dtype=np.uint8)

    colors, _, counts = _unique_colors(col_fit)
    colors, counts = _drop_off_curve(colors, counts)

    # Median reference level per colour, via a sort-and-split rather than a
    # Python loop over a few hundred groups on a megapixel image.
    lookup = {tuple(c): i for i, c in enumerate(colors.astype(np.uint8))}
    all_colors, all_inverse, _ = _unique_colors(col_fit)
    mapping = np.array([lookup.get(tuple(c), -1)
                        for c in all_colors.astype(np.uint8)], dtype=np.int64)
    idx = mapping[all_inverse.ravel()]

    keep = idx >= 0
    idx, levels = idx[keep], ref_fit[keep]
    order = np.argsort(idx, kind="stable")
    idx, levels = idx[order], levels[order]
    bounds = np.searchsorted(idx, np.arange(len(colors) + 1))
    medians = np.array([
        np.median(levels[bounds[i]:bounds[i + 1]]) if bounds[i + 1] > bounds[i]
        else np.nan for i in range(len(colors))])

    # How tightly does each colour pin down a loudness?  A real gradient entry
    # stands for one level, so the reference levels under it cluster; anything
    # painted over the top - a cursor, a selection edge, the ruled lines these
    # exports carry - lands at whatever loudness happens to be underneath, so
    # its levels are scattered.  Spread separates the two cleanly, and unlike a
    # rarity test it does not punish the genuinely rare bright end of a table.
    spreads = np.array([
        (np.subtract(*np.percentile(levels[bounds[i]:bounds[i + 1]], [75, 25]))
         if bounds[i + 1] - bounds[i] >= 8 else 0.0)
        for i in range(len(colors))])

    good = ~np.isnan(medians)
    if good.sum() >= 16:
        limit = max(0.05, 4.0 * float(np.median(spreads[good])))
        good &= spreads <= limit
    colors, medians, counts = colors[good], medians[good], counts[good]
    if len(colors) < 8:
        raise ValueError("too few colours matched between the two images")

    walked = colors[np.argsort(medians)]

    # Spread the ranked colours evenly over the table's 0..1 index.
    src = np.linspace(0.0, 1.0, len(walked))
    grid = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(grid, src, walked[:, c]) for c in range(3)], axis=1)
    return np.rint(np.clip(lut, 0, 255)).astype(np.uint8)
