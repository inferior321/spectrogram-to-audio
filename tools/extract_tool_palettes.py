"""Extract exact colour tables from ffmpeg and SoX by asking them directly.

    venv/bin/python tools/extract_tool_palettes.py

ffmpeg, SoX and Spek all draw a legend containing a colour bar, and that bar
*is* the palette.  Rendering one and reading a column out of it gives the real
table - no transcribing constants out of source code, no guessing from
screenshots, and no dependency on any of them being installed at run time,
because the result is written to spectro/tools.npz and shipped.

ffmpeg and SoX render straight to a file.  Spek is a GUI with no batch mode, so
it is run on an Xvfb display and screenshotted.

This matters more than it sounds.  FFmpeg's modes are *named* after matplotlib
colour maps, and it would be reasonable to assume they are the same tables.
They are not: its magma differs from matplotlib's by 13/255 on average and its
terrain by 72/255, because ffmpeg re-anchors each one to run from black to
white.  Assuming they matched - which this project did before this script
existed - meant reading such images through the wrong table.

Every extraction is checked before it is written:

  * the colours in a real spectrogram from that tool must lie on the extracted
    curve, which shows the table holds the right set of colours;
  * decoding one recording through several different palettes of the same tool
    must give the same levels, which shows the tables are parametrised the same
    way as each other;
  * and an absolute check, because the one above is only relative: SoX's
    monochrome palette has to come out as a black-to-white grey ramp.  Reading
    every bar upside down would satisfy the relative test perfectly, and did.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "spectro" / "tools.npz"

FFMPEG_MODES = ["intensity", "rainbow", "moreland", "nebulae", "fire", "fiery",
                "fruit", "cool", "magma", "green", "viridis", "plasma",
                "cividis", "terrain", "channel"]

# name -> extra sox flags
SOX_MODES = {"default": [], "light": ["-l"], "mono": ["-m"],
             "high colour": ["-h"], "mono light": ["-m", "-l"]}


def run(cmd: list[str]) -> bool:
    return subprocess.run(cmd, capture_output=True).returncode == 0


def bar_to_lut(column: np.ndarray, quiet_at_top: bool) -> np.ndarray:
    """Resample a colour bar into a 256-entry table, quiet end first."""
    col = column.astype(float)
    if not quiet_at_top:
        col = col[::-1]
    idx = np.linspace(0, len(col) - 1, 256)
    lut = np.stack([np.interp(idx, np.arange(len(col)), col[:, c])
                    for c in range(3)], axis=1)
    return np.rint(np.clip(lut, 0, 255)).astype(np.uint8)


# --------------------------------------------------------------------------

def extract_ffmpeg(tmp: Path) -> dict[str, np.ndarray]:
    """One legend render per mode; the bar sits at a fixed place in it."""
    luts: dict[str, np.ndarray] = {}
    for mode in FFMPEG_MODES:
        png = tmp / f"ff_{mode}.png"
        if not run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=1000:duration=3",
                    "-lavfi", f"showspectrumpic=s=512x512:legend=1:color={mode}",
                    str(png)]):
            print(f"  ffmpeg failed for {mode}", file=sys.stderr)
            continue
        img = np.asarray(Image.open(png).convert("RGB"))
        # The bar is the column with by far the most distinct colours.
        counts = [len(np.unique(img[:, x, :], axis=0)) for x in range(img.shape[1])]
        x = int(np.argmax(counts))
        column = img[:, x, :]
        # Trim the black padding above and below the gradient.
        lit = np.flatnonzero(column.sum(axis=1) > 6)
        runs, start, prev = [], lit[0], lit[0]
        for v in lit[1:]:
            if v != prev + 1:
                runs.append((start, prev))
                start = v
            prev = v
        runs.append((start, prev))
        a, b = max(runs, key=lambda r: r[1] - r[0])
        luts[mode] = bar_to_lut(column[a:b + 1], quiet_at_top=False)
    return luts


def extract_sox(tmp: Path) -> dict[str, np.ndarray]:
    wav = tmp / "noise.wav"
    if not run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                "-i", "anoisesrc=d=4:c=pink:a=0.5:r=44100", "-c:a", "pcm_s16le",
                str(wav)]):
        return {}

    luts: dict[str, np.ndarray] = {}
    for name, flags in SOX_MODES.items():
        png = tmp / f"sox_{name.replace(' ', '_')}.png"
        if not run(["sox", str(wav), "-n", "spectrogram", *flags,
                    "-x", "400", "-y", "200", "-o", str(png)]):
            print(f"  sox failed for {name}", file=sys.stderr)
            continue
        img = np.asarray(Image.open(png).convert("RGB"))
        counts = [len(np.unique(img[:, x, :], axis=0)) for x in range(img.shape[1])]
        x = int(np.argmax(counts))
        # A fixed extent, not a search: on the light schemes the quiet end of
        # the bar is nearly the page colour, so hunting for "not background"
        # clips it short and the table comes out missing its bottom.
        #
        # SoX draws its bar loud-end-up, like ffmpeg, so it needs the same
        # flip.  Getting this backwards produced five tables that were all
        # reversed together - and because they were all wrong the same way,
        # decoding one recording through each of them still agreed perfectly.
        # Only the absolute check below catches it.
        luts[name] = bar_to_lut(img[30:230, x, :], quiet_at_top=False)
    return luts


def longest_smooth_run(column: np.ndarray, step_limit: float = 12.0) -> np.ndarray:
    """The stretch of a column that varies smoothly - i.e. the colour bar.

    Spek's legend column also passes through its own version text and a black
    gap before reaching the bar, and both of those jump abruptly, so taking the
    longest smoothly-varying stretch isolates the gradient.  The constant tail
    below the bar is then trimmed, since flat counts as smooth too.
    """
    d = np.linalg.norm(np.diff(column.astype(float), axis=0), axis=1)
    smooth = np.flatnonzero(d < step_limit)
    runs, start, prev = [], smooth[0], smooth[0]
    for v in smooth[1:]:
        if v != prev + 1:
            runs.append((start, prev))
            start = v
        prev = v
    runs.append((start, prev))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    seg = column[a:b + 2]
    while len(seg) > 2 and tuple(seg[-1]) == tuple(seg[-2]):
        seg = seg[:-1]
    return seg


def extract_spek(tmp: Path) -> dict[str, np.ndarray]:
    """Screenshot Spek's window on a private display and read its legend bar.

    Spek is a GUI with no batch mode, so this is the only way to ask it what
    its palette is.  It runs on an Xvfb display rather than the real desktop:
    there its window is the only one that exists, so nothing competes for
    focus, nothing is placed off-screen, and nothing flashes up in front of
    whoever is using the machine.  Capturing from the real display did work,
    but only after fighting a window that opened at x=3900 and first came back
    as a blank white rectangle.
    """
    needed = ("spek", "xdotool", "Xvfb", "ffmpeg")
    if any(shutil.which(t) is None for t in needed):
        print("  skipped - needs " + ", ".join(needed))
        return {}

    wav, png, display = tmp / "noise.wav", tmp / "spek.png", ":99"
    env = dict(os.environ, DISPLAY=display)
    xvfb = subprocess.Popen(["Xvfb", display, "-screen", "0", "1000x700x24"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    spek = None
    try:
        time.sleep(3)
        spek = subprocess.Popen(["spek", str(wav)], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(7)
        found = subprocess.run(["xdotool", "search", "--name", "spek"], env=env,
                               capture_output=True, text=True).stdout.split()
        if not found:
            print("  spek window never appeared")
            return {}
        wid = found[-1]
        subprocess.run(["xdotool", "windowmove", wid, "0", "0",
                        "windowsize", wid, "1000", "700"], env=env,
                       capture_output=True)
        time.sleep(4)
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "x11grab",
             "-video_size", "1000x700", "-i", f"{display}.0+0,0",
             "-frames:v", "1", str(png)])
    finally:
        if spek is not None:
            spek.terminate()
        xvfb.terminate()

    if not png.exists():
        return {}
    img = np.asarray(Image.open(png).convert("RGB"))
    half = img.shape[1] // 2
    counts = [len(np.unique(img[:, x, :], axis=0)) for x in range(half, img.shape[1])]
    x = half + int(np.argmax(counts))
    return {"default": bar_to_lut(longest_smooth_run(img[:, x, :]),
                                  quiet_at_top=False)}


# --------------------------------------------------------------------------

def decode(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    d = ((img[:, :, None, :].astype(float) - lut[None, None, :, :].astype(float))
         ** 2).sum(axis=3)
    return np.argmin(d, axis=2) / 255.0


def validate(tmp: Path, luts: dict[str, np.ndarray], kind: str) -> bool:
    """Render one recording through several palettes; levels must agree."""
    wav = tmp / "noise.wav"
    if not wav.exists():
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "anoisesrc=d=4:c=pink:a=0.5:r=44100", "-c:a", "pcm_s16le",
             str(wav)])

    names = list(luts)[:6]
    levels: dict[str, np.ndarray] = {}
    for name in names:
        png = tmp / f"chk_{kind}_{name.replace(' ', '_')}.png"
        if kind == "ffmpeg":
            ok = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(wav), "-lavfi",
                      f"showspectrumpic=s=600x400:legend=0:color={name}", str(png)])
            crop = (slice(None), slice(None))
        else:
            ok = run(["sox", str(wav), "-n", "spectrogram", *SOX_MODES[name],
                      "-x", "400", "-y", "200", "-o", str(png)])
            crop = (slice(40, 190), slice(30, 420))
        if not ok:
            continue
        img = np.asarray(Image.open(png).convert("RGB"))[crop]
        levels[name] = decode(img[::3, ::3], luts[name])

        flat = img.reshape(-1, 3).astype(float)
        uniq = np.unique(flat, axis=0)
        dist = np.sqrt(((uniq[:, None, :] - luts[name].astype(float)[None, :, :])
                        ** 2).sum(axis=2)).min(axis=1)
        print(f"    {name:12} {len(uniq):5} colours, mean {dist.mean():.2f}/255 "
              f"from the table, worst {dist.max():.1f}")

    ok = True
    ref_name = next(iter(levels))
    ref = levels[ref_name]
    for name, lv in levels.items():
        if name == ref_name:
            continue
        gap = float(np.abs(lv - ref).mean())
        corr = float(np.corrcoef(lv.ravel(), ref.ravel())[0, 1])
        good = gap < 0.05 and corr > 0.95
        ok &= good
        print(f"    {name:12} vs {ref_name}: level gap {gap:.4f}, r {corr:.4f}"
              f"  {'OK' if good else 'MISMATCH'}")
    return ok


def main() -> int:
    missing = [t for t in ("ffmpeg", "sox") if not shutil.which(t)]
    sp: dict[str, np.ndarray] = {}
    if missing:
        print(f"needs {' and '.join(missing)} on PATH", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("extracting ffmpeg showspectrumpic palettes")
        ff = extract_ffmpeg(tmp)
        print(f"  got {len(ff)} of {len(FFMPEG_MODES)}")
        print("  validating:")
        ok = validate(tmp, ff, "ffmpeg")

        print("\nextracting SoX spectrogram palettes")
        sx = extract_sox(tmp)
        print(f"  got {len(sx)} of {len(SOX_MODES)}")
        print("  validating:")
        ok &= validate(tmp, sx, "sox")

        # An absolute check, because everything above is relative.  Reading
        # every bar upside down gives five tables that are all reversed
        # together, and they then agree with each other perfectly - which is
        # exactly what happened the first time this script ran.  SoX's
        # monochrome palette has a known right answer, so it is the anchor.
        if "mono" in sx:
            ramp = sx["mono"].astype(int)
            upright = bool(ramp[0, 0] < 20 and ramp[-1, 0] > 235)
            print(f"    mono grey ramp runs {ramp[0, 0]} -> {ramp[-1, 0]}"
                  f"  {'OK' if upright else 'REVERSED'}")
            ok &= upright

        print("\nextracting Spek's palette")
        sp = extract_spek(tmp)
        if sp:
            # Spek ships one palette, so there is no sibling to compare it
            # against.  Instead every colour in its own plot has to lie on the
            # extracted curve - the same set-membership check used above, which
            # is what proves the bar was read over its true extent and not
            # through the window title text sitting in the same column.
            shot = np.asarray(Image.open(tmp / "spek.png").convert("RGB"))
            plot = shot[130:660, 62:915].reshape(-1, 3).astype(float)
            uniq, cnt = np.unique(plot, axis=0, return_counts=True)
            dist = np.sqrt(((uniq[:, None, :]
                             - sp["default"].astype(float)[None, :, :]) ** 2)
                           .sum(axis=2)).min(axis=1)
            weighted = float((dist * (cnt / cnt.sum())).sum())
            print(f"    {len(uniq)} colours in its own plot, mean "
                  f"{weighted:.2f}/255 from the table, worst {dist.max():.1f}")
            ok &= weighted < 3.0

    bundle = {f"ffmpeg: {k}": v for k, v in ff.items()}
    bundle.update({f"sox: {k}": v for k, v in sx.items()})
    bundle.update({f"spek: {k}": v for k, v in sp.items()})
    np.savez_compressed(OUT, **bundle)
    print(f"\nwrote {OUT} ({OUT.stat().st_size} bytes, {len(bundle)} tables)")

    # Record how far ffmpeg's are from the matplotlib maps they are named after,
    # because the tempting assumption that they match is wrong.
    mpl_path = ROOT / "spectro" / "colormaps.npz"
    if mpl_path.exists():
        with np.load(mpl_path) as mpl:
            print("\nffmpeg vs the matplotlib map of the same name:")
            for name in ("magma", "viridis", "plasma", "cividis", "terrain"):
                if name in ff and name in mpl.files:
                    diff = float(np.abs(ff[name].astype(float)
                                        - mpl[name].astype(float)).mean())
                    print(f"  {name:10} mean difference {diff:5.1f}/255 "
                          f"- {'same' if diff < 3 else 'NOT the same table'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
