"""Derive Audacity's colour gradients from real exports.

Run this whenever you add new reference exports to testing-files/:

    venv/bin/python tools/derive_audacity_luts.py

It writes spectro/audacity.npz, which the app then treats as exact tables.

Expected files: one per scheme per frequency scale, named `<axis><n>.png`
where n is the scheme's position in Audacity's Scheme dropdown.  Having the
same scheme at two different frequency scales is what makes this trustworthy -
a colour table has nothing to do with the frequency axis, so the table derived
from `linear2.png` must equal the one derived from `mel2.png`.  If they differ,
the derivation is wrong, and that check needs no assumptions about the audio,
the settings, or the other schemes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectro import colormap, core  # noqa: E402
from spectro.colormap import LUM, learn_lut_from_pair  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "testing-files"

# Audacity's Scheme dropdown order -> the app's name for it.
SCHEMES = {
    1: "Audacity Color (default)",
    2: "Audacity Color (classic)",
    3: "Audacity Grayscale",
    4: "Audacity Inverse Grayscale",
}
# The greyscale schemes are exact by construction and are not learned.
EXACT = {
    "Audacity Grayscale": lambda r: np.stack([255 - r] * 3, axis=1),
    "Audacity Inverse Grayscale": lambda r: np.stack([r] * 3, axis=1),
}
REFERENCE_INDEX = 3          # the greyscale export used as the level reference
AXES = ("linear", "mel")


def image_path(axis: str, index: int) -> Path:
    return IMAGES / f"{axis}{'' if index == 1 else index}.png"


def load(axis: str, index: int) -> np.ndarray:
    return core.load_image(str(image_path(axis, index)))


def autocrop(rgb: np.ndarray, max_colors: int = 6, max_strip: int = 6) -> np.ndarray:
    """Strip the window chrome Audacity draws around the image.

    Testing for a *constant* row is not enough - these exports have a frame
    with a lighter highlight along one edge, so the outer rows hold three or
    four colours and survive a constant-row test.  Counting distinct colours
    separates them, but only up to a point, which is why the strip is capped:
    a quiet passage at the edge of a greyscale export is genuinely uniform, and
    without a cap this happily ate 32 columns of real audio from one image and
    2 from the next, leaving exports of the same sound no longer aligned with
    each other.  Chrome is a few pixels; anything more is data.
    """
    q = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    packed = (q[..., 0].astype(np.uint32) << 16 |
              q[..., 1].astype(np.uint32) << 8 | q[..., 2].astype(np.uint32))
    h, w = packed.shape

    def run(lines: np.ndarray) -> int:
        n = 0
        for i in range(min(max_strip, len(lines))):
            if len(np.unique(lines[i])) > max_colors:
                break
            n += 1
        return n

    t, b = run(packed), run(packed[::-1])
    l, r = run(packed.T), run(packed.T[::-1])
    return rgb[t:h - b, l:w - r]


def background(rgb: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Median colour around the edge - the quiet end of the gradient."""
    t = max(1, thickness)
    edge = np.concatenate([
        rgb[:t, :, :].reshape(-1, 3), rgb[-t:, :, :].reshape(-1, 3),
        rgb[:, :t, :].reshape(-1, 3), rgb[:, -t:, :].reshape(-1, 3),
    ])
    return np.median(edge, axis=0)


def coarse(x: np.ndarray, grid: tuple[int, int] = (256, 128)) -> np.ndarray:
    """Area-average a level image down, so comparisons survive misalignment.

    Two exports of the same audio are rendered at slightly different pixel
    sizes, and spectrogram detail is fine enough that a one-pixel shift wrecks
    a pixel-wise correlation even when the two images agree about the sound.
    """
    img = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.resize(grid, Image.BOX), dtype=np.float32) / 255.0


def lut_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(float) - b.astype(float)).mean())


def main() -> int:
    missing = [image_path(ax, i) for ax in AXES for i in SCHEMES
               if not image_path(ax, i).is_file()]
    if missing:
        print("missing reference exports:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    ramp = np.arange(256, dtype=np.uint8)
    luts: dict[str, np.ndarray] = {}
    per_axis: dict[str, dict[str, np.ndarray]] = {ax: {} for ax in AXES}
    ok = True

    # ---- derive each colour scheme, once per frequency scale --------------
    print("deriving colour tables")
    for index, name in SCHEMES.items():
        if name in EXACT:
            luts[name] = EXACT[name](ramp)
            print(f"  {name:32} exact by construction")
            continue
        for axis in AXES:
            # Paired with the greyscale export of the SAME axis: that image
            # states outright how loud each point was, so the colours can be
            # ranked by loudness instead of guessed at from their geometry.
            per_axis[axis][name] = learn_lut_from_pair(
                autocrop(load(axis, index)),
                autocrop(load(axis, REFERENCE_INDEX)),
                gray_light_is_quiet=True)

    # ---- cross-check: the two axes must agree ----------------------------
    # This is the strong test.  A colour table is a property of the scheme
    # alone, so deriving it from a linear export and from a mel export of the
    # same scheme must give the same answer; the two images share nothing but
    # that scheme.
    print("\ncross-check - same scheme derived from linear vs mel")
    for index, name in SCHEMES.items():
        if name in EXACT:
            continue
        a, b = per_axis["linear"][name], per_axis["mel"][name]
        diff = lut_distance(a, b)
        verdict = "OK" if diff < 12.0 else "DISAGREE"
        if verdict != "OK":
            ok = False
        print(f"  {name:32} mean difference {diff:5.1f}/255  {verdict}")
        # Average the two: each is an independent estimate of the same table.
        luts[name] = np.rint((a.astype(float) + b.astype(float)) / 2).astype(np.uint8)

    out = ROOT / "spectro" / "audacity.npz"
    np.savez_compressed(out, **luts)
    colormap._load_bundled.cache_clear()
    print(f"\nwrote {out} ({out.stat().st_size} bytes)")
    for name in SCHEMES.values():
        lut = luts[name]
        stops = ", ".join(f"{i / 255:.2f}:{tuple(int(v) for v in lut[i])}"
                          for i in (0, 85, 170, 255))
        print(f"  {name:32} {stops}")

    # ---- validate: every scheme must recover the same levels -------------
    # Now that the exports are sorted by frequency axis, a colour export and
    # the greyscale export of the same axis show the same picture, so their
    # recovered level images can be compared directly.
    print("\nvalidation - recovered levels vs the greyscale export, per axis")
    for axis in AXES:
        ref_rgb = autocrop(load(axis, REFERENCE_INDEX))
        ref = coarse(colormap.level_from_lut(
            ref_rgb, luts[SCHEMES[REFERENCE_INDEX]]))
        print(f"  {axis}:")
        for index, name in SCHEMES.items():
            if index == REFERENCE_INDEX:
                continue
            rgb = autocrop(load(axis, index))
            got = coarse(colormap.level_from_lut(rgb, luts[name]))
            corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
            # Different exports may use different Gain/Range, which shifts and
            # scales level but cannot reorder it, so correlation is the fair
            # measure and the absolute difference is only reported.
            mae = float(np.abs(got - ref).mean())
            verdict = "OK" if corr > 0.90 else "SUSPECT"
            if verdict != "OK":
                ok = False
            print(f"    {name:32} r {corr:.4f}  (mean level gap {mae:.3f})  {verdict}")

    # ---- sanity: the two axes must NOT match each other -------------------
    # If linear and mel exports scored the same, the files would be mislabelled
    # and none of the above would mean what it claims.
    print("\nsanity - linear and mel really are different pictures")
    ref_lin = coarse(colormap.level_from_lut(
        autocrop(load("linear", REFERENCE_INDEX)), luts[SCHEMES[REFERENCE_INDEX]]))
    ref_mel = coarse(colormap.level_from_lut(
        autocrop(load("mel", REFERENCE_INDEX)), luts[SCHEMES[REFERENCE_INDEX]]))
    cross = float(np.corrcoef(ref_lin.ravel(), ref_mel.ravel())[0, 1])
    print(f"  greyscale linear vs greyscale mel: r {cross:.4f} "
          f"({'as expected - different axes' if cross < 0.9 else 'TOO SIMILAR'})")
    if cross >= 0.9:
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
