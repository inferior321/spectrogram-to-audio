"""Correctness checks.

The important one is `roundtrip`: build audio whose content is known, draw it
into a PNG the way Audacity draws one, read that PNG back and check the
recovered spectrum matches.  It verifies the dB mapping, the axis mapping and
the resynthesis together - a sign error or an off-by-one in any of them shows
up as a large spectral error.

Run with:  venv/bin/python -m tests.test_roundtrip
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectro import core  # noqa: E402
from spectro.settings import FREQ_SCALES, WINDOW_TYPES, Settings  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------

def render_like_audacity(x: np.ndarray, st: Settings) -> np.ndarray:
    """Draw `x` as a greyscale spectrogram image using Audacity's rules."""
    window = core.make_window(st.window_type, st.window_size)
    spec = np.abs(core.stft(x, st, window))
    # Audacity normalises by the window sum so a full-scale sine reads 0 dB.
    spec = spec / (np.sum(window) / 2.0)
    db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    level = np.clip((db + st.range_db + st.gain_db) / st.range_db, 0.0, 1.0)

    # Resample FFT bins onto image rows for the chosen scale.
    h = 754
    unit = np.linspace(0.0, 1.0, h)
    lo, hi = core._scale_forward(np.array([st.min_freq, st.max_freq]), st.freq_scale)
    grid = np.linspace(max(st.min_freq, 1e-3), st.max_freq, 8192)
    row_freqs = np.interp(lo + unit * (hi - lo),
                          core._scale_forward(grid, st.freq_scale), grid)
    bin_freqs = np.fft.rfftfreq(st.n_fft, d=1.0 / st.sample_rate)
    rows = np.array([np.interp(row_freqs, bin_freqs, level[:, t])
                     for t in range(level.shape[1])]).T

    img = np.flipud(rows)                        # top row = highest frequency
    grey = np.clip((1.0 - img) * 255.0, 0, 255).astype(np.uint8)   # invert
    return np.stack([grey] * 3, axis=2)


def test_roundtrip() -> None:
    print("\nRound trip: audio -> Audacity-style PNG -> audio")
    st = Settings()
    st.gl_iterations = 60
    st.normalize = False       # so the ABSOLUTE dB calibration is under test
    sr = st.sample_rate
    dur = 3.0
    t = np.arange(int(sr * dur)) / sr

    # Three steady tones plus noise.  The whole signal is scaled to sit inside
    # the window the image can actually represent (Gain 20 / Range 80 means
    # -100..-20 dBFS): anything louder is clipped to black when drawn and
    # cannot come back, which would be a property of the picture rather than a
    # fault in this code.
    tones = [440.0, 1500.0, 6000.0]
    x = sum(0.3 * np.sin(2 * np.pi * f * t) for f in tones)
    x += 0.01 * np.random.default_rng(1).standard_normal(len(t))
    x = x / np.max(np.abs(x)) * (10 ** (-15 / 20))

    rgb = render_like_audacity(x, st).astype(np.float32) / 255.0
    st.duration_s = dur
    res = core.convert(core.to_level(rgb, st), st)

    check("duration preserved", abs(res.duration_s - dur) < 0.15,
          f"{res.duration_s:.3f}s vs {dur}s")

    # Long-term spectra, on the same normalisation the picture uses.
    w = core.make_window("Hann", st.window_size)
    scale = np.sum(w) / 2.0

    def spectrum_db(sig: np.ndarray) -> np.ndarray:
        s = np.abs(core.stft(sig, st, w)).mean(axis=1) / scale
        return 20.0 * np.log10(np.maximum(s, 1e-12))

    din, dout = spectrum_db(x), spectrum_db(res.audio)
    freqs = np.fft.rfftfreq(st.n_fft, 1.0 / sr)

    # The magnitude we BUILD from the picture is deterministic - level mapping,
    # axis mapping and window scaling - so it gets a tight tolerance.  What
    # Griffin-Lim then ACHIEVES is a non-convex search that always lands a few
    # dB low, because phases that are not mutually consistent partly cancel
    # when the frames are overlap-added.  Testing the two separately is what
    # keeps this suite able to catch a real mapping error.
    target_db = 20.0 * np.log10(
        np.maximum(core.build_magnitude(core.to_level(rgb, st), st).mean(axis=1)
                   / scale, 1e-12))

    for f in tones:
        k = int(np.argmin(np.abs(freqs - f)))
        band = slice(max(0, k - 3), k + 4)
        ref, tgt, got = din[band].max(), target_db[band].max(), dout[band].max()
        check(f"tone {f:.0f} Hz: built magnitude within 2.5 dB",
              abs(ref - tgt) < 2.5, f"reference {ref:.1f} -> built {tgt:.1f} dB")
        check(f"tone {f:.0f} Hz: resynthesis within 6 dB",
              abs(ref - got) < 6.0, f"reference {ref:.1f} -> output {got:.1f} dB")

    quiet = (freqs > 9000) & (freqs < 19000)
    check("no spurious high-frequency energy",
          dout[quiet].max() < din[quiet].max() + 10.0,
          f"in {din[quiet].max():.1f} dB, out {dout[quiet].max():.1f} dB")

    # Compare in the level domain, which is what a picture can hold: the mean
    # absolute error is in units where 1.0 is the entire 80 dB range.
    band = (freqs >= 50) & (freqs <= st.max_freq * 0.95)
    lin = np.clip((din[band] + st.range_db + st.gain_db) / st.range_db, 0, 1)
    lout = np.clip((dout[band] + st.range_db + st.gain_db) / st.range_db, 0, 1)
    mae = float(np.mean(np.abs(lin - lout)))
    corr = float(np.corrcoef(lin, lout)[0, 1])
    check("mean level error under 0.05 (4 dB of 80)", mae < 0.05,
          f"MAE {mae:.4f} = {mae * st.range_db:.1f} dB")
    check("level correlation > 0.95", corr > 0.95, f"r = {corr:.3f}")


def test_level_mapping() -> None:
    print("\nLevel mapping matches Audacity's formula")
    st = Settings(gain_db=20, range_db=80)
    freqs = np.array([1000.0])
    for level, expect_db in [(1.0, -20.0), (0.5, -60.0), (0.25, -80.0)]:
        mag = core.level_to_magnitude(np.array([[level]], dtype=np.float32), freqs, st)
        got = 20 * np.log10(mag[0, 0])
        check(f"level {level} -> {expect_db} dBFS", abs(got - expect_db) < 0.01,
              f"got {got:.2f}")
    mag = core.level_to_magnitude(np.array([[0.0]], dtype=np.float32), freqs, st)
    check("level 0 is exact silence", mag[0, 0] == 0.0)


def test_scales() -> None:
    print("\nFrequency scales are monotonic and invert correctly")
    f = np.linspace(20, 20000, 500)
    for scale in FREQ_SCALES:
        s = core._scale_forward(f, scale)
        check(f"{scale} monotonic", bool(np.all(np.diff(s) > 0)))
        u = core.freq_to_unit(np.array([20.0, 20000.0]), 20, 20000, scale)
        check(f"{scale} endpoints map to 0 and 1",
              abs(u[0]) < 1e-9 and abs(u[1] - 1) < 1e-9)


def test_windows_and_stft() -> None:
    print("\nSTFT/ISTFT reconstruct a signal for every window type")
    st = Settings(window_size=512, zero_padding=1, overlap=4)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(512 * 20)
    for wt in WINDOW_TYPES:
        w = core.make_window(wt, st.window_size)
        spec = core.stft(x, st, w)
        y = core.istft(spec, st, w, len(x))
        # Ignore the edges, where overlap-add has no neighbours to sum with.
        a, b = st.window_size, len(x) - st.window_size
        err = np.max(np.abs(y[a:b] - x[a:b]))
        check(f"{wt} perfect reconstruction", err < 1e-6, f"max err {err:.2e}")


def test_time_control() -> None:
    print("\nDuration and stretch behave")
    st = Settings()
    st.duration_s = 10.0
    n = core.infer_frame_count(1703, st)
    check("explicit duration honoured", abs(core.frames_to_duration(n, st) - 10.0) < 0.05,
          f"{core.frames_to_duration(n, st):.3f}s")
    st.time_stretch = 2.0
    n2 = core.infer_frame_count(1703, st)
    check("stretch x2 doubles length",
          abs(core.frames_to_duration(n2, st) - 20.0) < 0.05)
    st.duration_s = 0.0
    st.time_stretch = 1.0
    n3 = core.infer_frame_count(1703, st)
    check("blank duration infers 1 column = 1 frame", n3 == 1703, f"{n3} frames")


def test_no_edge_burst() -> None:
    """Regression: the overlap-add edges must not produce a loud transient.

    The first and last window's worth of samples are covered by fewer frames
    than the middle.  Left alone, that produced a spike tens of dB above the
    real content, which then captured the peak normaliser and pushed the whole
    file far too quiet - so this checks the edges are no louder than the body.
    """
    print("\nNo burst at the start or end")
    st = Settings(gl_iterations=24)
    rng = np.random.default_rng(7)
    # A broadband, fairly even image: any large transient must be an artefact.
    level = np.clip(0.45 + 0.12 * rng.standard_normal((256, 500)), 0, 1).astype(np.float32)
    res = core.convert(level, st)
    a = res.audio

    w = st.window_size
    if len(a) > 4 * w:
        head, tail, body = a[:w], a[-w:], a[w:-w]
        body_peak = float(np.abs(body).max())
        check("start is not louder than the body",
              float(np.abs(head).max()) <= body_peak * 1.5,
              f"head {np.abs(head).max():.4f} vs body {body_peak:.4f}")
        check("end is not louder than the body",
              float(np.abs(tail).max()) <= body_peak * 1.5,
              f"tail {np.abs(tail).max():.4f} vs body {body_peak:.4f}")

    rms = float(np.sqrt((a ** 2).mean()))
    crest = 20 * np.log10(float(np.abs(a).max()) / max(rms, 1e-12))
    check("crest factor is plausible for audio (< 26 dB)", crest < 26.0,
          f"{crest:.1f} dB")


def test_edge_cases() -> None:
    print("\nEdge cases")
    st = Settings(gl_iterations=2)
    black = np.zeros((32, 40, 3), dtype=np.float32)      # all silence once inverted
    res = core.convert(core.to_level(black + 1.0, st), st)
    check("all-white image gives silence", float(np.max(np.abs(res.audio))) == 0.0)

    tiny = np.zeros((4, 4, 3), dtype=np.float32)
    res = core.convert(core.to_level(tiny, st), st)
    check("4x4 image does not crash", len(res.audio) > 0)

    st2 = Settings(gl_iterations=2, crop_left=1000, crop_top=1000)
    lev = core.to_level(np.zeros((50, 50, 3), dtype=np.float32), st2)
    check("absurd crop is clamped, not fatal", lev.size > 0, f"shape {lev.shape}")

    st3 = Settings(gl_iterations=2, max_freq=48000, sample_rate=8000)
    res = core.convert(core.to_level(np.zeros((20, 30, 3), dtype=np.float32), st3), st3)
    check("max_freq above Nyquist is handled", np.all(np.isfinite(res.audio)))


def test_image_formats(tmp: Path) -> None:
    print("\nImage loading")
    tmp.mkdir(parents=True, exist_ok=True)
    arr = (np.random.default_rng(2).random((40, 60, 3)) * 255).astype(np.uint8)
    for ext in ("png", "jpg", "bmp", "webp"):
        p = tmp / f"t.{ext}"
        Image.fromarray(arr).save(p)
        got = core.load_image(str(p))
        check(f"{ext} loads", got.shape == (40, 60, 3))

    p = tmp / "rgba.png"
    rgba = np.dstack([arr, np.full((40, 60), 128, np.uint8)])
    Image.fromarray(rgba, "RGBA").save(p)
    check("RGBA flattens to RGB", core.load_image(str(p)).shape == (40, 60, 3))

    p = tmp / "grey.png"
    Image.fromarray(arr[:, :, 0], "L").save(p)
    check("greyscale L loads", core.load_image(str(p)).shape == (40, 60, 3))




# ==========================================================================
# Colour schemes, denoise, preview and PGHI
# ==========================================================================

def test_colour_schemes() -> None:
    """Every bundled gradient must invert back to the level that drew it."""
    print("\nColour schemes round-trip")
    from spectro import colormap
    rng = np.random.default_rng(3)
    lvl = np.clip(rng.beta(2, 5, (120, 200)), 0, 1).astype(np.float32)
    worst = ("", 0.0)
    for name in colormap.available():
        lut = colormap.get_lut(name)
        if lut is None:
            continue
        img = lut[np.rint(lvl * 255).astype(int)].astype(np.float32) / 255.0
        back = colormap.level_from_scheme(img, name)
        mae = float(np.abs(back - lvl).mean())
        if mae > worst[1]:
            worst = (name, mae)
    check("every bundled table inverts exactly", worst[1] < 0.01,
          f"worst was {worst[0]} at MAE {worst[1]:.5f}")


def test_table_fit() -> None:
    """A table must explain its own colours and reject everyone else's.

    Nothing in the program identifies schemes any more - you say which one
    drew the image - so this is no longer about detection.  It is about the
    tables being right: if the wrong table fitted nearly as well, then reading
    an image through the scheme you picked would not mean much.
    """
    print("\nColour tables fit their own images")
    from spectro import colormap
    rng = np.random.default_rng(4)
    lvl = np.clip(rng.beta(2, 5, (150, 260)), 0, 1)
    names = ["matplotlib: viridis", "matplotlib: magma", "matplotlib: jet",
             "matplotlib: turbo", "Audacity Color (default)",
             "Audacity Color (classic)"]
    for name in names:
        lut = colormap.get_lut(name)
        img = lut[np.rint(lvl * 255).astype(int)].astype(np.float32) / 255.0
        own = colormap.fit_error(img, name)
        others = [colormap.fit_error(img, other) for other in names if other != name]
        check(f"{name}: fits its own image", own < 1.0, f"{own:.3f}/255")
        check(f"{name}: wrong tables fit far worse", min(others) > 10 * max(own, 0.05),
              f"own {own:.2f} vs best other {min(others):.2f}")


def test_denoise() -> None:
    """Denoise must lower the floor without touching the peaks."""
    print("\nDenoise")
    rng = np.random.default_rng(6)
    mag = np.abs(rng.standard_normal((200, 300))).astype(np.float32) * 0.01
    mag[50:60, 100:150] += 1.0                      # a loud event
    st = Settings(denoise_db=18)
    # The floor is now MEASURED from a marked patch rather than guessed as a
    # percentile, so the test supplies one: the median of each bin's quiet part.
    floor = np.median(mag[:, :90], axis=1)
    out = core.denoise(mag, st, floor)
    floor_before = float(np.percentile(mag, 20))
    floor_after = float(np.percentile(out, 20))
    check("floor comes down", floor_after < floor_before * 0.3,
          f"{floor_before:.5f} -> {floor_after:.5f}")
    check("peak is preserved", abs(float(out.max()) - float(mag.max())) < 0.02,
          f"{mag.max():.3f} -> {out.max():.3f}")
    # denoise_db = 0 must be a true no-op, so leaving the control alone can
    # never change a result: checked through build_magnitude, which is what
    # actually decides whether denoise runs at all.
    rng2 = np.random.default_rng(11)
    level = np.clip(rng2.beta(2, 5, (80, 120)), 0, 1).astype(np.float32)
    prof = level.mean(axis=1)
    off = core.build_magnitude(level, Settings(denoise_db=0.0, gl_iterations=1),
                               noise=prof)
    on = core.build_magnitude(level, Settings(denoise_db=12.0, gl_iterations=1),
                              noise=prof)
    check("denoise 0 dB is a true no-op",
          np.array_equal(off, core.build_magnitude(
              level, Settings(denoise_db=0.0, gl_iterations=1), noise=prof)))
    check("denoise 12 dB does change the magnitudes", not np.array_equal(off, on))
    # And with no profile it must do nothing, whatever the dB is set to.
    check("denoise with no profile is a no-op",
          np.array_equal(
              core.build_magnitude(level, Settings(denoise_db=0.0, gl_iterations=1)),
              core.build_magnitude(level, Settings(denoise_db=30.0, gl_iterations=1))))


def test_preview_region_shortens_the_audio() -> None:
    """A time region must shorten the OUTPUT, not just crop the picture.

    The duration you type describes the whole image, so a region covering a
    twentieth of it is a twentieth of that duration.  Without this the region
    cropped the columns and then stretched them over the full length: on a
    3:34 import a six per cent selection still synthesised 214 seconds from 102
    pixels, which made the preview slower than converting the whole file and
    wrong as well.
    """
    print("\nPreview region shortens the audio")
    st = Settings(duration_s=3 * 60 + 34)
    width = 1703
    whole = core.infer_frame_count(width, st)

    st6 = st.copy()
    st6.preview_start, st6.preview_end = 0.30, 0.36
    part = core.infer_frame_count(int(width * 0.06), st6)

    check("a 6% region asks for about 6% of the frames",
          0.04 < part / whole < 0.08, f"{part} of {whole} = {part / whole * 100:.1f}%")
    check("and about 6% of the seconds",
          abs(core.frames_to_duration(part, st6) - 214 * 0.06) < 2.0,
          f"{core.frames_to_duration(part, st6):.1f}s of 214s")

    # The inferred-duration path must scale too, and the whole image is 100%.
    st_full = st.copy()
    st_full.preview_start, st_full.preview_end = 0.0, 1.0
    check("no region means the full length",
          core.infer_frame_count(width, st_full) == whole)
    check("preview_fraction is 1 when the region is the whole width",
          core.preview_fraction(st_full) == 1.0)
    check("preview_fraction is 1 when the region is degenerate",
          core.preview_fraction(Settings(preview_start=0.5, preview_end=0.5)) == 1.0)


def test_orientation() -> None:
    """A rotated picture must be readable, and reading one wrongly must show.

    Some tools can draw the spectrogram turned on its side - ffmpeg's
    showspectrum will, and Spectrogrammer exposes it as an Orientation setting.
    Read as though it were the normal way up, frequency becomes time and the
    result is noise rather than something slightly off, so this checks both
    that the setting rescues it and that the mistake is loud.
    """
    print("\nImage orientation")
    from spectro.settings import ORIENTATIONS
    st = Settings()
    rgb = render_like_audacity(
        np.sin(2 * np.pi * 1000 * np.arange(int(st.sample_rate * 2.0))
               / st.sample_rate) * 0.2, st).astype(np.float32) / 255.0

    check("normal layout leaves the image alone",
          core.apply_orientation(rgb, Settings(orientation=ORIENTATIONS[0])).shape == rgb.shape)

    for name in ORIENTATIONS[1:]:
        s2 = Settings(orientation=name)
        turned = core.apply_orientation(rgb, s2)
        check(f"{name}: swaps the axes",
              turned.shape[0] == rgb.shape[1] and turned.shape[1] == rgb.shape[0],
              f"{rgb.shape[1]}x{rgb.shape[0]} -> {turned.shape[1]}x{turned.shape[0]}")

    # Rotate the picture, then read it back with the matching setting: the
    # level image must come out the same as the unrotated one did.
    want = core.to_level(rgb, Settings())
    turned = np.ascontiguousarray(np.rot90(rgb, k=1))
    s3 = Settings(orientation="Time on Y, rotate clockwise")
    got = core.to_level(core.apply_orientation(turned, s3), s3)
    check("a rotated image reads back identically with the right setting",
          got.shape == want.shape and float(np.abs(got - want).max()) < 1e-6,
          f"{got.shape} vs {want.shape}")

    # And reading it wrongly must be obviously wrong, not subtly so.
    wrong = core.to_level(core.apply_orientation(turned, Settings()), Settings())
    check("reading a rotated image as normal is obviously wrong",
          wrong.shape != want.shape, f"{wrong.shape} vs {want.shape}")


def test_preview_region() -> None:
    """The preview must cut the image, and only in time."""
    print("\nPreview region")
    rgb = np.zeros((60, 400, 3), dtype=np.float32)
    st = Settings(preview_start=0.25, preview_end=0.5)
    lev = core.to_level(rgb, st)
    check("columns are sliced", lev.shape[1] == 100, f"{lev.shape[1]} of 400")
    check("rows are untouched", lev.shape[0] == 60, f"{lev.shape[0]}")
    full = core.to_level(rgb, Settings())
    check("full range is the whole image", full.shape[1] == 400)
    # A region of exactly zero width means "no region", and gives everything.
    empty = core.to_level(rgb, Settings(preview_start=0.5, preview_end=0.5))
    check("a zero-width region means the whole image", empty.shape[1] == 400,
          f"{empty.shape[1]} of 400")

    # A region that is tiny but real is honoured, floored at a few columns.
    # It used to fall back to the whole image, which is the worst answer
    # available: asking for a sliver and being given three and a half minutes.
    tiny = core.to_level(rgb, Settings(preview_start=0.5, preview_end=0.5005))
    check("a sliver is honoured, not silently widened to everything",
          4 <= tiny.shape[1] <= 8, f"{tiny.shape[1]} columns")


def test_pghi() -> None:
    """PGHI must beat random phase, which is the only reason it exists."""
    print("\nPGHI phase estimate")
    st = Settings(zero_padding=2)
    win = core.make_window(st.window_type, st.window_size)
    sr = st.sample_rate
    t = np.arange(int(sr * 2.0)) / sr
    rng = np.random.default_rng(7)
    x = sum(0.3 * np.sin(2 * np.pi * f * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
            for f in (220.0, 440.0, 880.0))
    x = x + 0.02 * rng.standard_normal(len(t))
    x = x / np.abs(x).max()
    mag = np.abs(core.stft(x, st, win)).astype(np.float32)
    length = core.raw_length(mag.shape[1], st)

    def converge(angles: np.ndarray) -> float:
        y = core.istft(mag * angles, st, win, length)
        got = np.abs(core.stft(y, st, win))
        n = min(got.shape[1], mag.shape[1])
        return float(np.linalg.norm(got[:, :n] - mag[:, :n])
                     / np.linalg.norm(mag[:, :n]))

    rand = converge(np.exp(2j * np.pi * rng.random(mag.shape)))
    pghi = converge(np.exp(1j * core.pghi_phase(mag, st)))
    check("PGHI beats random phase before any iteration", pghi < rand * 0.5,
          f"random {rand:.4f} vs PGHI {pghi:.4f}")


def test_window_should_match_the_export() -> None:
    """The resynthesis window belongs to the image, not to taste.

    A Gaussian window sounds smoother - it has no sidelobes, so there is less
    smeared energy for the phase reconstruction to invent a wrong phase for -
    and the fit error the program reports internally gets *better* when you
    choose one.  That number is self-referential: it scores the output against
    the target through the same window, so a wider-lobed window is marked
    against a blurrier target and flatters itself.

    Measured against the original audio instead, a mismatched window is clearly
    worse.  This test exists so that the tempting change - defaulting to a
    Gaussian because it sounds nicer - cannot be made without the number that
    contradicts it showing up.
    """
    print("\nResynthesis window vs the original audio")
    st0 = Settings(zero_padding=1)
    sr, dur = st0.sample_rate, 3.0
    t = np.arange(int(sr * dur)) / sr
    rng = np.random.default_rng(3)
    tones = [300.0, 900.0, 2500.0, 6000.0]
    x = sum(0.25 * np.sin(2 * np.pi * f * t) * (1 + 0.4 * np.sin(2 * np.pi * 2.5 * t))
            for f in tones)
    x = x + 0.01 * rng.standard_normal(len(t))
    x = x / np.abs(x).max() * (10 ** (-15 / 20))
    rgb = render_like_audacity(x, st0).astype(np.float32) / 255.0   # Hann-drawn

    ref_w = core.make_window("Hann", st0.window_size)

    def spec_db(sig: np.ndarray) -> np.ndarray:
        s = np.abs(core.stft(sig, st0, ref_w)).mean(axis=1) / (np.sum(ref_w) / 2)
        return 20 * np.log10(np.maximum(s, 1e-12))

    ref = spec_db(x)
    freqs = np.fft.rfftfreq(st0.n_fft, 1.0 / sr)
    band = (freqs >= 100) & (freqs <= 15000)

    errors = {}
    for wt in ("Hann", "Gaussian(a=4.5)"):
        s = st0.copy()
        s.window_type, s.gl_iterations, s.duration_s = wt, 60, dur
        s.normalize = False
        out = spec_db(core.convert(core.to_level(rgb, s), s).audio)
        errors[wt] = float(np.abs(out[band] - ref[band]).mean())

    check("matching the export's window is the most accurate",
          errors["Hann"] < errors["Gaussian(a=4.5)"],
          f"Hann {errors['Hann']:.2f} dB vs Gaussian {errors['Gaussian(a=4.5)']:.2f} dB")
    check("and by a wide margin",
          errors["Gaussian(a=4.5)"] > 2 * errors["Hann"],
          f"{errors['Gaussian(a=4.5)'] / errors['Hann']:.1f}x worse")


def test_pghi_gamma() -> None:
    """The Gaussian time-frequency constants must be derived, not guessed.

    These were originally guessed at 0.245 and 0.19; the truth is pi/(2a^2),
    which is 0.128 and 0.078.  Being about twice too large made PGHI's phase
    estimate worse than starting from noise - on precisely the windows the
    method is derived for.  The check on Hann and Blackman is the other half:
    those carry published constants, and a sweep on true magnitudes puts the
    optimum on top of them, which is what says the surrounding formula is
    right rather than the constants absorbing an error in it.
    """
    print("\nPGHI window constants")
    for alpha in (3.5, 4.5):
        name = f"Gaussian(a={alpha})"
        want = np.pi / (2 * alpha ** 2)
        got = core._GAMMA_C[name]
        check(f"{name} constant is pi/(2a^2)", abs(got - want) < 1e-9,
              f"{got:.5f} vs {want:.5f}")

    check("Hann keeps its published constant",
          abs(core._GAMMA_C["Hann"] - 0.25645) < 1e-6)

    # On a true spectrogram - the case the theory covers - PGHI with a Gaussian
    # window must beat random phase outright.  With the guessed constant it lost.
    st = Settings(window_type="Gaussian(a=4.5)", zero_padding=1)
    w = core.make_window(st.window_type, st.window_size)
    sr = st.sample_rate
    t = np.arange(int(sr * 2.0)) / sr
    rng = np.random.default_rng(5)
    x = sum(0.3 * np.sin(2 * np.pi * f * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
            for f in (220.0, 440.0, 880.0))
    x = (x + 0.02 * rng.standard_normal(len(t)))
    x = x / np.abs(x).max()
    mag = np.abs(core.stft(x, st, w)).astype(np.float32)
    length = core.raw_length(mag.shape[1], st)

    def conv(angles: np.ndarray) -> float:
        y = core.istft(mag * angles, st, w, length)
        got = np.abs(core.stft(y, st, w))
        n = min(got.shape[1], mag.shape[1])
        return float(np.linalg.norm(got[:, :n] - mag[:, :n])
                     / np.linalg.norm(mag[:, :n]))

    rand = conv(np.exp(2j * np.pi * rng.random(mag.shape)))
    pghi = conv(np.exp(1j * core.pghi_phase(mag, st)))
    check("PGHI with a Gaussian window beats random by 5x", pghi < rand / 5,
          f"random {rand:.4f} vs PGHI {pghi:.4f}")


def test_audacity_tables() -> None:
    """The derived tables must let a level be read back unambiguously.

    An earlier version of this test demanded that a table's two ends be far
    apart and that it never double back in RGB space.  Both demands were wrong:
    Audacity's Color (classic) genuinely runs pale blue-white -> blue ->
    magenta -> red -> pale pink-white, so it really does fold, and its ends
    really are close.  What actually matters is weaker and more useful - that
    two levels which are far apart never share a colour, because that is what
    makes the table invertible.
    """
    print("\nDerived Audacity tables")
    from spectro import colormap
    for name in ("Audacity Color (default)", "Audacity Color (classic)"):
        lut = colormap.get_lut(name)
        if lut is None:
            continue
        table = lut.astype(np.float64)
        d = np.sqrt(((table[:, None, :] - table[None, :, :]) ** 2).sum(axis=2))
        far = np.abs(np.subtract.outer(np.arange(256), np.arange(256))) > 64
        worst = float(d[far].min())
        check(f"{name}: distant levels never share a colour", worst > 10.0,
              f"closest such pair is {worst:.0f}/255 apart")

        # Neighbouring entries should differ, or the table has flat stretches
        # where a range of levels cannot be told apart at all.
        steps = np.linalg.norm(np.diff(table, axis=0), axis=1)
        check(f"{name}: no flat stretch", float(np.mean(steps < 0.5)) < 0.10,
              f"{np.mean(steps < 0.5) * 100:.0f}% of steps are flat")


def test_source_catalogue() -> None:
    """Every catalogue entry must name a gradient that actually exists."""
    print("\nSource catalogue")
    from spectro import colormap, sources

    entries = [x for x in sources.catalogue() if not x.is_separator]
    check("catalogue is populated", len(entries) >= 30, f"{len(entries)} entries")

    broken = [x.label for x in entries
              if x.scheme not in colormap.available()]
    check("every entry points at a real gradient", not broken, str(broken[:3]))

    labelled = [x.label for x in entries if x.accuracy not in ("exact", "generic")]
    check("every entry states its accuracy", not labelled, str(labelled[:3]))

    # Audacity's four and the matplotlib set must all be marked exact, since
    # those tables are derived or lifted rather than guessed.
    inexact = [x.label for x in entries
               if (x.label.startswith("Audacity") or x.label.startswith("matplotlib"))
               and x.accuracy != "exact"]
    check("Audacity and matplotlib entries are marked exact", not inexact,
          str(inexact[:3]))

    # Nothing in the catalogue may be a borrowed gradient: an entry named after
    # a tool has to carry that tool's real table, or not be offered at all.
    approx = [x.label for x in entries if x.accuracy not in ("exact", "generic")]
    check("no unverified entries remain", not approx, str(approx))

    named = [x for x in entries if x.accuracy == "exact"]
    check("every exact entry has a real table",
          all(colormap.get_lut(x.scheme) is not None for x in named),
          f"{len(named)} exact entries")

    check("round-trips through label_for_scheme",
          sources.label_for_scheme("Audacity Color (classic)")
          == "Audacity — Color (classic)")


def test_tool_palettes() -> None:
    """The ffmpeg and SoX tables must be present, distinct, and not matplotlib's.

    The last part is the one worth asserting.  FFmpeg names several modes after
    matplotlib colour maps, and this project once assumed they were the same
    tables; they are not, and an image read through the wrong one comes out
    with warped levels.
    """
    print("\nffmpeg and SoX palettes")
    from spectro import colormap
    names = colormap.available()

    ff = [n for n in names if n.startswith("ffmpeg: ")]
    sox = [n for n in names if n.startswith("sox: ")]
    spek = [n for n in names if n.startswith("spek: ")]
    check("ffmpeg palettes are bundled", len(ff) == 15, f"{len(ff)} of 15")
    check("SoX palettes are bundled", len(sox) == 5, f"{len(sox)} of 5")
    check("Spek palette is bundled", len(spek) == 1, f"{len(spek)} of 1")

    # Spek runs black -> purple -> red -> orange -> cream, so its ends are far
    # apart and it gets brighter throughout; a table read upside down or over
    # the wrong extent would not.
    sp = colormap.get_lut("spek: default")
    if sp is not None:
        lum = sp.astype(float) @ np.array([0.299, 0.587, 0.114])
        check("spek runs dark to light", lum[0] < 20 and lum[-1] > 230,
              f"{lum[0]:.0f} .. {lum[-1]:.0f}")
        check("spek brightens monotonically overall",
              float(np.mean(np.diff(lum) >= -2)) > 0.95,
              f"{np.mean(np.diff(lum) >= -2) * 100:.0f}% of steps")

    # SoX's monochrome mode is a plain grey ramp, which is checkable outright.
    mono = colormap.get_lut("sox: mono")
    if mono is not None:
        chan = mono.astype(int)
        check("sox mono is neutral grey",
              int(np.abs(chan[:, 0] - chan[:, 1]).max()) <= 3
              and int(np.abs(chan[:, 1] - chan[:, 2]).max()) <= 3)
        check("sox mono runs dark to light",
              chan[0, 0] < 20 and chan[-1, 0] > 235,
              f"{chan[0, 0]} .. {chan[-1, 0]}")

    for name in ("magma", "viridis", "plasma", "cividis", "terrain"):
        a = colormap.get_lut(f"ffmpeg: {name}")
        b = colormap.get_lut(f"matplotlib: {name}")
        if a is None or b is None:
            continue
        diff = float(np.abs(a.astype(float) - b.astype(float)).mean())
        check(f"ffmpeg {name} is NOT matplotlib {name}", diff > 5.0,
              f"they differ by {diff:.1f}/255")


def test_window_does_not_grow(tmp: Path) -> None:
    """Loading an image must never change how much room the window asks for.

    A QLabel reports its pixmap's size as its size hint, and the image view
    rescales its pixmap to whatever size it currently has, so the two used to
    feed each other: a wide image made the widget ask to be wider, which made
    the pixmap wider, which grew the window - and it never shrank back.

    The assertion is on the size HINT rather than the window size, because a
    window manager is what actually enforces the growth; offscreen the window
    stays put either way, and a test that only watched its size would have
    passed throughout the bug.
    """
    print("\nWindow does not resize itself")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
        from spectro.gui import MainWindow
    except Exception as exc:                       # pragma: no cover
        print(f"  SKIP - no Qt available ({exc})")
        return

    tmp.mkdir(parents=True, exist_ok=True)
    wide = tmp / "wide.png"
    tall = tmp / "tall.png"
    Image.fromarray((np.random.default_rng(0).random((200, 5000, 3)) * 255)
                    .astype(np.uint8)).save(wide)
    Image.fromarray((np.random.default_rng(1).random((3000, 400, 3)) * 255)
                    .astype(np.uint8)).save(tall)

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        win.resize(1000, 700)
        win.show()
        app.processEvents()
        before = win.image_view.sizeHint()
        floor = win.minimumSizeHint()

        for path in (wide, tall, wide):
            win.load_image(path)
            app.processEvents()
            now = win.image_view.sizeHint()
            check(f"{path.name}: the view's size hint is unchanged",
                  now == before, f"{before.width()}x{before.height()} -> "
                                 f"{now.width()}x{now.height()}")
            grew = win.minimumSizeHint()
            check(f"{path.name}: the window's minimum is unchanged",
                  grew.width() <= floor.width() and grew.height() <= floor.height(),
                  f"{floor.width()}x{floor.height()} -> "
                  f"{grew.width()}x{grew.height()}")
    finally:
        win.close()
        app.processEvents()


def test_zoom_and_pan(tmp: Path) -> None:
    """Zoom must magnify without moving the window or losing the mapping.

    Everything the view reports - the frequency and time under the pointer, the
    edges of a dragged region - is a coordinate in the image, so it all has to
    survive being magnified and slid about.  The awkward one is zooming about
    the pointer: the image point under the cursor has to stay under the cursor,
    or the picture jumps away as you zoom into it.
    """
    print("\nZoom and pan")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
        from spectro.gui import MainWindow
    except Exception as exc:                       # pragma: no cover
        print(f"  SKIP - no Qt available ({exc})")
        return

    tmp.mkdir(parents=True, exist_ok=True)
    wide = tmp / "zoomable.png"
    Image.fromarray((np.random.default_rng(2).random((300, 4000, 3)) * 255)
                    .astype(np.uint8)).save(wide)

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        win.resize(1100, 760)
        win.show()
        app.processEvents()
        win.load_image(wide)
        app.processEvents()
        view = win.image_view
        hint, size = view.sizeHint(), (win.width(), win.height())

        cx, cy = view.width() / 2, view.height() / 2
        u, v = view.to_unit(cx, cy)
        check("at fit, the view centre is the image centre",
              abs(u - 0.5) < 0.01 and abs(v - 0.5) < 0.01, f"({u:.3f}, {v:.3f})")

        # Zooming about a point must leave that point where it was.
        spot = (view.width() * 0.25, view.height() * 0.5)
        before = view.to_unit(*spot)
        view.zoom_by(3.0, spot)
        after = view.to_unit(*spot)
        check("zooming about the pointer keeps that point still",
              abs(before[0] - after[0]) < 0.005 and abs(before[1] - after[1]) < 0.005,
              f"{before[0]:.3f} -> {after[0]:.3f}")
        check("zoom actually changed", abs(view.zoom_level() - 3.0) < 1e-6,
              f"{view.zoom_level()}")

        check("the window did not move", (win.width(), win.height()) == size,
              f"{size} -> {(win.width(), win.height())}")
        check("the size hint did not change", view.sizeHint() == hint)

        # Panning cannot wander off the image.
        view._cx = -9.0
        view._clamp()
        left = view._cx
        view._cx = 9.0
        view._clamp()
        right = view._cx
        check("panning is clamped to the image", 0.0 < left < right < 1.0,
              f"{left:.3f} .. {right:.3f}")

        # A region is stored in image coordinates, so zoom must not disturb it.
        view.reset_view()
        view.set_selection((0.40, 0.46))
        view.zoom_by(5.0, (view.width() * 0.4, cy))
        check("a marked region survives zooming",
              view.selection() == (0.40, 0.46), str(view.selection()))

        check("zoom will not go below fit",
              (view.reset_view(), view.zoom_by(0.1), view.zoom_level())[2] == 1.0,
              f"{view.zoom_level()}")
        view.zoom_by(1e6)
        check("zoom is capped", view.zoom_level() <= view.MAX_ZOOM,
              f"{view.zoom_level()}")

        # Loading another image starts from fit again.
        win.load_image(wide)
        app.processEvents()
        check("a new image starts at fit", view.zoom_level() == 1.0)
    finally:
        win.close()
        app.processEvents()


class _FakePlayer:
    """Stands in for the audio device: records calls, makes no sound."""

    available = True
    error = None

    def __init__(self) -> None:
        self.playing = False
        self.calls: list[str] = []

    def play(self, audio, sample_rate) -> None:
        self.playing = True
        self.calls.append(f"play:{len(audio) / sample_rate:.2f}")

    def stop(self) -> None:
        if self.playing:
            self.calls.append("stop")
        self.playing = False

    def is_playing(self) -> bool:
        return self.playing


def test_preview_replaces_playback() -> None:
    """Starting a render must silence whatever is already playing.

    Two faults used to combine here.  Nothing stopped playback when a new
    render began, so the previous preview carried on underneath it; and the
    finished handler called the Play/Stop *toggle*, which - finding audio still
    playing - switched it off.  The old clip therefore cut out at the very
    moment the new one should have started, and the new one was never heard at
    all.
    """
    print("\nPreview replaces what is playing")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
        from spectro.gui import MainWindow
    except Exception as exc:                       # pragma: no cover
        print(f"  SKIP - no Qt available ({exc})")
        return

    root = Path(__file__).resolve().parent.parent / "testing-files"
    sample = root / "linear3.png"
    if not sample.is_file():
        print("  SKIP - reference exports not present")
        return

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        win.player = _FakePlayer()
        win.show()
        win.load_image(sample)
        win.cmb_quality.setCurrentText("Draft - fastest preview")
        win._apply_quality()

        def run(region: tuple[float, float]) -> None:
            win.image_view.set_selection(region)
            win.start_preview()
            deadline = time.time() + 300
            while win._thread is not None and time.time() < deadline:
                app.processEvents()
                time.sleep(0.02)

        run((0.10, 0.16))
        check("the first preview plays", win.player.is_playing(),
              str(win.player.calls))
        first = [c for c in win.player.calls if c.startswith("play:")][0]

        run((0.50, 0.60))
        calls = win.player.calls
        check("the second preview is playing too", win.player.is_playing(), str(calls))
        check("the previous one was stopped first", "stop" in calls, str(calls))
        played = [c for c in calls if c.startswith("play:")]
        check("the SECOND clip is what plays, not the first",
              len(played) == 2 and played[1] != first, str(played))
        check("stop came between the two", calls.index("stop") < len(calls) - 1,
              str(calls))

        # And the button is still a toggle for the user.
        win.toggle_play()
        check("the Play button stops it", not win.player.is_playing())
        win.toggle_play()
        check("and starts it again", win.player.is_playing())
    finally:
        win.close()
        app.processEvents()


def test_bin_ratio_advice() -> None:
    """A mismatched FFT must name the setting that fixes it.

    The window size is set by the image's HEIGHT - the transform is twice the
    frequency axis - and it is natural to reach for the width instead, which on
    a 32768x2048 export is sixteen times too big. The readout used to answer
    every mismatch with "try a smaller zero padding", which says nothing when
    the padding is already 1 and the window is what is wrong.
    """
    print("\nBin-ratio advice")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
        from spectro.gui import MainWindow
    except Exception as exc:                       # pragma: no cover
        print(f"  SKIP - no Qt available ({exc})")
        return

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        # A 2048-row image wants an FFT of 4096.
        win.rgb = np.zeros((2048, 4000, 3), dtype=np.float32)
        for size, padding, want in (("4096", "1", "well matched"),
                                    ("2048", "2", "well matched"),
                                    ("32768", "1", "try window 4096"),
                                    ("32768", "2", "try window 2048"),
                                    ("512", "1", "try window 4096")):
            win.cmb_window_size.setCurrentText(size)
            win.cmb_padding.setCurrentText(padding)
            text = win.lbl_bins.text()
            check(f"window {size} padding {padding} -> '{want}'", want in text, text)

        win.cmb_window_size.setCurrentText("32768")
        win.cmb_padding.setCurrentText("1")
        check("it no longer blames the padding when padding is 1",
              "zero padding" not in win.lbl_bins.text(), win.lbl_bins.text())
    finally:
        win.close()
        app.processEvents()


def test_calibration_names_the_right_ends() -> None:
    """The dB readout must name the colours the chosen scheme actually uses.

    It used to say "white = quiet, black = loud" whatever was selected. That is
    right for Audacity's Grayscale and backwards for ffmpeg, SoX, Spek and
    Audacity's own Inverse Grayscale, all of which run black to white - so the
    readout told half the users the opposite of the truth.
    """
    print("\nCalibration readout")
    from spectro import colormap

    for scheme, quiet, loud in (
            ("Audacity Grayscale", "white", "black"),
            ("Audacity Inverse Grayscale", "black", "white"),
            ("ffmpeg: intensity", "black", "white"),
            ("sox: mono", "black", "white"),
            ("sox: light", "white", "black"),
            ("Greyscale (light = quiet)", "white", "black"),
            ("Greyscale (dark = quiet)", "black", "white")):
        got = colormap.scheme_ends(scheme)
        check(f"{scheme}: {quiet} -> {loud}", got == (quiet, loud), str(got))

    # A scheme whose ends are nearly the same colour must still distinguish
    # them, since that is the interesting thing about it.
    q, l = colormap.scheme_ends("Audacity Color (classic)")
    check("Color (classic) tells its two pale ends apart", q != l, f"{q} / {l}")

    # And the readout itself must use them, not the words white and black.
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
        from spectro.gui import MainWindow
    except Exception as exc:                       # pragma: no cover
        print(f"  SKIP - no Qt for the readout ({exc})")
        return
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        labels = [win.cmb_source.itemText(i) for i in range(win.cmb_source.count())]
        index = labels.index("ffmpeg — intensity")
        win.cmb_source.setCurrentIndex(index)
        win._on_source_changed(index)
        text = win.lbl_calib.text().lower()
        check("a black-to-white scheme reads that way round",
              text.index("black") < text.index("white"), win.lbl_calib.text())
    finally:
        win.close()
        app.processEvents()


def test_readouts_agree_about_dbfs() -> None:
    """Hover, calibration line and conversion must quote the same dB.

    Each of the three carried its own copy of the Audacity formula
    `level*range - range - gain`. With Plain dB range selected, hovering an
    unlit pixel read "-100.0 dBFS" while the calibration line an inch below it
    said the quiet end was -80, and the conversion used a third value again.
    They all go through core.level_to_db now.
    """
    print("\ndBFS readouts")
    from spectro import core
    from spectro.settings import Settings

    for mapping, lo, hi in (("Audacity dB (gain/range)", -100.0, -20.0),
                            ("Plain dB range", -80.0, 0.0)):
        st = Settings()
        st.level_mapping, st.range_db, st.gain_db = mapping, 80.0, 20.0
        ends = (core.level_to_db(0.0, st, silent_at_zero=False),
                core.level_to_db(1.0, st, silent_at_zero=False))
        check(f"{mapping}: ends are {lo} .. {hi}",
              abs(ends[0] - lo) < 0.01 and abs(ends[1] - hi) < 0.01, str(ends))

        # And the middle of the ramp must match what the conversion will do.
        for lvl in (0.25, 0.5, 0.75):
            mag = core.level_to_magnitude(np.array([[lvl]], dtype=np.float32),
                                          np.array([1000.0]), st)[0, 0]
            said = core.level_to_db(lvl, st)
            check(f"{mapping}: level {lvl} agrees with the conversion",
                  abs(20 * np.log10(mag) - said) < 0.01,
                  f"conversion {20 * np.log10(mag):.2f} vs readout {said:.2f}")

    # An unlit pixel is silence, not the bottom of the scale.
    st = Settings()
    check("level 0 reads as silent, not as the floor",
          core.level_to_db(0.0, st) == float("-inf"))


def test_ffmpeg_sources_are_calibrated() -> None:
    """ffmpeg entries must pin the mapping the way SoX and Spek do.

    showspectrumpic documents drange (default 120 dBFS) and limit (default 0),
    so brightness maps as dB = (level - 1) * drange - there is nothing to
    guess. The entries carried no overrides, so an ffmpeg export was read with
    the Audacity dialog's 80 dB range and 20 dB gain: 40 dB of stretch in the
    wrong direction, and a 1 Hz bottom on an axis that starts at 0.
    """
    print("\nffmpeg calibration")
    from spectro import sources

    seen = 0
    for src in sources.catalogue():
        if not src.label.startswith("ffmpeg —"):
            continue
        seen += 1
        ov = src.overrides
        check(f"{src.label} pins the mapping",
              ov.get("level_mapping") == "Plain dB range"
              and ov.get("range_db") == 120.0
              and ov.get("min_freq") == 0.0, str(ov))
    check("every ffmpeg mode was checked", seen >= 10, f"{seen} modes")


def test_crop_preview() -> None:
    """The picture on screen must be the pixels that get converted.

    Crop used to be invisible: the sliders were connected to nothing, so the
    preview showed the whole image however much was trimmed, and there was no
    way to see whether a border had actually been removed.
    """
    print("\nCrop preview")
    from spectro.settings import Settings

    # The clamp: opposing crops must never leave an empty array, whatever is
    # asked for. Every stage downstream divides by the size.
    for ct, cb, cl, cr in ((0, 0, 0, 0), (5, 5, 7, 3), (60, 60, 0, 0),
                           (999, 999, 999, 999), (-4, -4, -4, -4),
                           (49, 50, 0, 0), (0, 0, 79, 80)):
        st = Settings()
        st.crop_top, st.crop_bottom = ct, cb
        st.crop_left, st.crop_right = cl, cr
        rgb = np.zeros((50, 80, 3), dtype=np.float32)
        view = core.crop_view(rgb, st)
        ok = view.shape[0] >= 1 and view.shape[1] >= 1 and view.size > 0
        check(f"crop t{ct} b{cb} l{cl} r{cr} leaves something",
              ok, f"{view.shape[1]} x {view.shape[0]}")

    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
        from spectro.gui import MainWindow
    except Exception as exc:                       # pragma: no cover
        print(f"  SKIP - no Qt ({exc})")
        return
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        rng = np.random.default_rng(3)
        win.raw_rgb = rng.random((120, 200, 3)).astype(np.float32)
        win.rgb = win.raw_rgb.copy()
        win._show_cropped()
        check("uncropped preview is the whole image",
              win.image_view._pixmap.width() == 200
              and win.image_view._pixmap.height() == 120,
              f"{win.image_view._pixmap.width()} x {win.image_view._pixmap.height()}")

        # Drive the spin boxes, which is what a user edit actually moves:
        # set_value() guards its signals on purpose, so it proves nothing here.
        win.sld_crop_l.spin.setValue(20); win.sld_crop_r.spin.setValue(10)
        win.sld_crop_t.spin.setValue(5);  win.sld_crop_b.spin.setValue(15)
        check("the preview shrank to the cropped size",
              win.image_view._pixmap.width() == 170
              and win.image_view._pixmap.height() == 100,
              f"{win.image_view._pixmap.width()} x {win.image_view._pixmap.height()}")

        # It must be the RIGHT pixels, not merely the right size.
        shown = win.cropped_rgb()
        check("the preview shows the correct region",
              np.array_equal(shown, win.raw_rgb[5:105, 20:190]))

        # And the conversion must read exactly what is displayed.
        lvl = core.to_level(win.rgb, win.current_settings())
        check("the conversion reads the same region",
              lvl.shape[1] == shown.shape[1] and lvl.shape[0] == shown.shape[0],
              f"level {lvl.shape[1]} x {lvl.shape[0]} vs shown "
              f"{shown.shape[1]} x {shown.shape[0]}")

        # Zoom must survive a crop change, or trimming an edge zoomed in is
        # impossible - which is when the zoom is most wanted.
        win.image_view.zoom_by(4.0)
        z = win.image_view.zoom_level()
        win.sld_crop_l.set_value(25)
        check("zoom survives a crop change",
              abs(win.image_view.zoom_level() - z) < 1e-9,
              f"{z} -> {win.image_view.zoom_level()}")

        # Reset sets the sliders from code, and SliderSpin.set_value suppresses
        # valueChanged - so the picture has to be refreshed explicitly or it
        # keeps showing the crop that was just cleared.
        win.reset_defaults()
        check("Reset restores the full picture",
              win.image_view._pixmap.width() == 200
              and win.image_view._pixmap.height() == 120,
              f"{win.image_view._pixmap.width()} x {win.image_view._pixmap.height()}")

        # Loading a fresh image should still reset the view.
        win.image_view.zoom_by(3.0)
        win.image_view.set_array(win.raw_rgb)
        check("a new image resets the zoom",
              abs(win.image_view.zoom_level() - 1.0) < 1e-9,
              f"zoom {win.image_view.zoom_level()}")
    finally:
        win.close()
        app.processEvents()


def test_image_controls_wait_for_an_image() -> None:
    """Settings that describe the picture stay greyed out until there is one.

    The crop sliders were the giveaway: their range is a quarter of the image,
    so with nothing loaded they offered a meaningless 0-400 and moving them did
    nothing anyone could see. Output preferences stay live throughout - format,
    quality and sample rate are worth setting before loading anything.
    """
    print("\nControls gated on a loaded image")
    import os
    import tempfile
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
        from spectro.gui import MainWindow
    except Exception as exc:                       # pragma: no cover
        print(f"  SKIP - no Qt ({exc})")
        return

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    # isEnabled() reports the EFFECTIVE state, so a disabled section shows up
    # on every control inside it.
    gated = {"Came from": win.cmb_source, "Layout": win.cmb_orientation,
             "Flip polarity": win.chk_flip, "Crop left": win.sld_crop_l,
             "Crop bottom": win.sld_crop_b, "Scale": win.cmb_scale,
             "Min frequency": win.sld_fmin, "Max frequency": win.sld_fmax,
             "Mapping": win.cmb_mapping, "Gain": win.sld_gain,
             "Range": win.sld_range, "Noise gate": win.sld_gate}
    live = {"Window size": win.cmb_window_size, "Overlap": win.cmb_overlap,
            "Sample rate": win.cmb_rate, "Duration": win.edit_duration,
            "Quality": win.cmb_quality, "Iterations": win.sld_iters,
            "Reduce background": win.sld_denoise, "Format": win.cmb_format,
            "Fade in/out": win.sld_fade}
    try:
        for name, wdg in gated.items():
            check(f"{name} is disabled before an image", not wdg.isEnabled())
        for name, wdg in live.items():
            check(f"{name} stays usable before an image", wdg.isEnabled())
        check("Convert is disabled before an image", not win.btn_convert.isEnabled())

        # Reset must not switch them on - there is still no image.
        win.reset_defaults()
        check("Reset does not enable them while nothing is loaded",
              not win.sld_crop_l.isEnabled() and not win.cmb_source.isEnabled())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.png"
            rng = np.random.default_rng(11)
            arr = (rng.random((64, 96, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(path)
            win.load_image(path)

        for name, wdg in gated.items():
            check(f"{name} is enabled once an image is loaded", wdg.isEnabled())
        check("Convert is enabled once an image is loaded", win.btn_convert.isEnabled())
        # Crop ranges are now the image's, not the placeholder 0-400.
        from spectro.gui import _crop_limit
        check("crop sliders re-range to the image",
              win.sld_crop_l._max == _crop_limit(96)
              and win.sld_crop_t._max == _crop_limit(64),
              f"left max {win.sld_crop_l._max} (want {_crop_limit(96)}), "
              f"top max {win.sld_crop_t._max} (want {_crop_limit(64)})")
    finally:
        win.close()
        app.processEvents()


def test_pitch() -> None:
    """Pitch must scale the frequency axis exactly and leave the length alone.

    The shift happens in build_magnitude, before phase reconstruction, so it is
    deterministic and can be checked exactly - measuring the finished audio
    instead would be reading Griffin-Lim's noise.
    """
    print("\nPitch")
    h, w = 2000, 260

    def built_centroid(pitch: float, drawn: float) -> float:
        st = Settings()
        st.min_freq, st.max_freq = 0.0, 22050.0
        st.pitch = pitch
        u = core.freq_to_unit(np.array([drawn]), st.min_freq, st.max_freq,
                              st.freq_scale)[0]
        row = int(round((1.0 - u) * (h - 1)))
        # Audacity Grayscale: the white page is silence, the black line is loud.
        rgb = np.ones((h, w, 3), dtype=np.float32)
        rgb[row - 1:row + 2] = 0.0
        mag = core.build_magnitude(core.to_level(rgb, st), st)
        freqs = np.fft.rfftfreq(st.n_fft, 1.0 / st.sample_rate)
        prof = mag.mean(axis=1)
        return float(np.sum(freqs * prof) / np.sum(prof))

    for pitch in (0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
        got, want = built_centroid(pitch, 3000.0), 3000.0 * pitch
        check(f"{pitch:.2f}x puts 3000 Hz at {want:.0f} Hz",
              abs(got - want) / want < 0.01, f"got {got:.1f} Hz")

    # The whole axis has to move, not one favoured line.
    for drawn in (500.0, 8000.0):
        got, want = built_centroid(2.0, drawn), 2.0 * drawn
        check(f"2.00x moves {drawn:.0f} Hz to {want:.0f} Hz",
              abs(got - want) / want < 0.01, f"got {got:.1f} Hz")

    # Length must not move: that is what separates this from Sample rate.
    rng = np.random.default_rng(5)
    rgb = rng.random((128, 300, 3)).astype(np.float32)
    lengths = []
    for pitch in (0.5, 1.0, 2.0):
        st = Settings()
        st.pitch, st.gl_iterations, st.duration_s = pitch, 3, 4.0
        lengths.append(len(core.convert(core.to_level(rgb, st), st).audio))
    check("pitch does not change the length", len(set(lengths)) == 1,
          f"lengths {lengths}")

    # 1.0 must be a true no-op, not merely close.
    st_a, st_b = Settings(), Settings()
    st_b.pitch = 1.0
    a = core.build_magnitude(core.to_level(rgb, st_a), st_a)
    b = core.build_magnitude(core.to_level(rgb, st_b), st_b)
    check("pitch 1.0 changes nothing at all", np.array_equal(a, b))

    # A nonsense value must not divide by zero or invert the axis.
    for bad in (0.0, -1.0):
        st = Settings()
        st.pitch = bad
        out = core.build_magnitude(core.to_level(rgb, st), st)
        check(f"pitch {bad} falls back to 1.0",
              np.array_equal(out, a), "differs from unpitched")


def test_crop_limit() -> None:
    """One crop slider must reach at least 100 px where the image allows.

    A quarter of the image capped a 288-pixel-tall matplotlib figure at 72,
    which is not far enough to trim past a legend.
    """
    print("\nCrop slider range")
    from spectro.gui import _crop_limit

    for size, want in ((288, 100), (432, 108), (2048, 512), (200, 100)):
        got = _crop_limit(size)
        check(f"a {size} px side allows {want} px", got == want, f"got {got}")
    # Tiny images must still leave something behind.
    for size in (1, 2, 5, 50):
        got = _crop_limit(size)
        check(f"a {size} px side stays under the image", 1 <= got <= max(1, size - 1),
              f"limit {got}")


def test_noise_profile() -> None:
    """Marking a quiet stretch must subtract that floor from the whole image."""
    print("\nNoise profile")
    rng = np.random.default_rng(9)
    h, w = 256, 400
    # A picture that is hiss everywhere, plus a loud band in the second half.
    lvl = np.clip(0.30 + 0.02 * rng.standard_normal((h, w)), 0, 1)
    lvl[120:140, 200:] = 0.95
    rgb = np.repeat((1.0 - lvl)[:, :, None], 3, axis=2).astype(np.float32)  # white=quiet

    st = Settings()
    st.denoise_db, st.noise_start, st.noise_end = 24.0, 0.0, 0.25
    prof = core.noise_profile(rgb, st)
    check("a marked region yields one number per row",
          prof is not None and prof.shape == (h,),
          "None" if prof is None else str(prof.shape))
    check("the profile reads the hiss level, not silence",
          0.2 < float(np.mean(prof)) < 0.45, f"mean {float(np.mean(prof)):.3f}")

    st_off = Settings()
    st_off.denoise_db = 0.0
    plain = core.build_magnitude(core.to_level(rgb, st_off), st_off)
    cut = core.build_magnitude(core.to_level(rgb, st), st, noise=prof)

    quiet = plain[:, :150]
    quiet_cut = cut[:, :150]
    check("the background comes down",
          float(quiet_cut.mean()) < float(quiet.mean()) * 0.4,
          f"{float(quiet.mean()):.4g} -> {float(quiet_cut.mean()):.4g}")
    check("the loud band survives",
          float(cut.max()) > float(plain.max()) * 0.7,
          f"{float(plain.max()):.4g} -> {float(cut.max()):.4g}")

    # No region marked means no profile and no change at all.
    st_none = Settings()
    st_none.denoise_db = 24.0
    check("nothing marked yields no profile", core.noise_profile(rgb, st_none) is None)
    untouched = core.build_magnitude(core.to_level(rgb, st_none), st_none,
                                     noise=core.noise_profile(rgb, st_none))
    check("nothing marked leaves the magnitudes alone",
          np.array_equal(untouched, plain))

    # Frequency smoothing must not change the overall amount of cut much, only
    # how abruptly it varies between neighbouring bins.
    st_rough, st_smooth = st.copy(), st.copy()
    st_rough.denoise_smoothing, st_smooth.denoise_smoothing = 0, 8
    rough = core.build_magnitude(core.to_level(rgb, st_rough), st_rough, noise=prof)
    smooth = core.build_magnitude(core.to_level(rgb, st_smooth), st_smooth, noise=prof)
    rough_jag = float(np.abs(np.diff(rough, axis=0)).mean())
    smooth_jag = float(np.abs(np.diff(smooth, axis=0)).mean())
    check("smoothing reduces bin-to-bin jaggedness", smooth_jag < rough_jag,
          f"{rough_jag:.4g} -> {smooth_jag:.4g}")

    # Sensitivity: lower cuts harder.
    outs = []
    for sens in (0.5, 1.0, 2.0):
        s2 = st.copy()
        s2.denoise_sensitivity = sens
        outs.append(float(core.build_magnitude(
            core.to_level(rgb, s2), s2, noise=prof)[:, :150].mean()))
    check("higher sensitivity cuts harder", outs[0] > outs[1] > outs[2],
          f"0.5x={outs[0]:.4g}  1.0x={outs[1]:.4g}  2.0x={outs[2]:.4g}")

    # The profile must come from the whole image, not the previewed slice:
    # marking the FIRST quarter must still work while previewing the LAST.
    st_far = st.copy()
    st_far.preview_start, st_far.preview_end = 0.75, 1.0
    far = core.noise_profile(rgb, st_far)
    check("the noise patch need not be inside the preview region",
          far is not None and np.allclose(far, prof, atol=0.02))


def test_text_is_readable() -> None:
    """Every label must contrast with the background it sits on.

    The hint text under each control used to be styled palette(mid). That is a
    BORDER colour: nothing requires it to contrast with the window, and on a
    dark theme it landed 7 luminance units from the background - a faint
    outline where a sentence should be. The default text colour is the only one
    the theme guarantees to be readable against its own background, so the
    hints now set only their size and inherit the rest.
    """
    print("\nLabel contrast")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtGui import QPalette
        from PyQt6.QtWidgets import QApplication, QLabel
        from spectro.gui import MainWindow
        from spectro.widgets import ImageView
    except Exception as exc:                       # pragma: no cover
        print(f"  SKIP - no Qt available ({exc})")
        return

    def lum(c) -> float:
        return 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        background = lum(app.palette().color(QPalette.ColorRole.Window))
        worst = (None, 1e9)
        for label in win.findChildren(QLabel):
            if isinstance(label, ImageView):
                continue          # paints its own dark background
            # Ask for the ACTIVE group explicitly. Sections that describe the
            # picture start disabled, and a disabled control is meant to be
            # dim - measuring its greyed colour would fail the test for doing
            # exactly the right thing. What matters is that the text is
            # readable when the control is usable.
            colour = label.palette().color(QPalette.ColorGroup.Active,
                                           label.foregroundRole())
            contrast = abs(lum(colour) - background)
            if contrast < worst[1]:
                worst = (label.objectName() or (label.text()[:30] or "<blank>"), contrast)
        check("no label is washed into the background", worst[1] > 60,
              f"worst is {worst[1]:.1f} on '{worst[0]}'")

        # And the specific mistake, named, so it cannot come back by that route.
        offenders = [n for n in ("lbl_file", "lbl_hover", "lbl_source",
                                 "lbl_calib", "lbl_bins", "lbl_time")
                     if "palette(mid)" in getattr(win, n).styleSheet()]
        check("no hint label uses palette(mid) as a text colour", not offenders,
              str(offenders))
    finally:
        win.close()
        app.processEvents()


def test_gui_source_selection() -> None:
    """The dropdown must set the scheme, and loading must not override it."""
    print("\nGUI source selection")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
        from spectro.gui import MainWindow
    except Exception as exc:                       # pragma: no cover
        print(f"  SKIP - no Qt available ({exc})")
        return

    root = Path(__file__).resolve().parent.parent / "testing-files"
    sample = root / "linear2.png"
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        from spectro import sources
        labels = [win.cmb_source.itemText(i) for i in range(win.cmb_source.count())]
        headings = [i for i in range(win.cmb_source.count())
                    if not win.cmb_source.model().item(i).isEnabled()]
        check("headings are present and unselectable", len(headings) >= 4,
              f"{len(headings)} headings")

        for label, want in (("Audacity — Color (classic)", "Audacity Color (classic)"),
                            ("matplotlib — turbo", "matplotlib: turbo"),
                            ("Spek", "spek: default")):
            index = labels.index(label)
            win.cmb_source.setCurrentIndex(index)
            win._on_source_changed(index)
            check(f"choosing '{label}' selects its gradient",
                  win.current_settings().color_scheme == want,
                  f"got {win.current_settings().color_scheme}")

        # The "any picture" entry also switches off the dB curve.
        index = labels.index("Any picture → sound (no dB curve)")
        win.cmb_source.setCurrentIndex(index)
        win._on_source_changed(index)
        check("the sonify entry drops the dB curve",
              win.current_settings().level_mapping == "Linear amplitude",
              win.current_settings().level_mapping)

        if sample.is_file():
            before = win.current_settings().color_scheme
            win.load_image(sample)
            check("loading an image changes nothing",
                  win.current_settings().color_scheme == before,
                  f"{before} -> {win.current_settings().color_scheme}")

        # A source may pin the level mapping too, and switching away must undo
        # exactly that and nothing else.
        labels = [win.cmb_source.itemText(i) for i in range(win.cmb_source.count())]
        if "SoX — default" in labels:
            index = labels.index("SoX — default")
            win.cmb_source.setCurrentIndex(index)
            win._on_source_changed(index)
            st = win.current_settings()
            check("SoX applies its documented 120 dB calibration",
                  st.level_mapping == "Plain dB range" and st.range_db == 120,
                  f"{st.level_mapping}, {st.range_db}")

            index = labels.index("matplotlib — viridis")
            win.cmb_source.setCurrentIndex(index)
            win._on_source_changed(index)
            st = win.current_settings()
            check("switching away undoes it",
                  st.level_mapping == Settings().level_mapping
                  and st.range_db == Settings().range_db,
                  f"{st.level_mapping}, {st.range_db}")

            # A value the user set by hand is not a source's to reset.
            win.sld_gain.slider.setValue(win.sld_gain.slider.value() + 10)
            by_hand = win.current_settings().gain_db
            index = labels.index("ffmpeg — magma")
            win.cmb_source.setCurrentIndex(index)
            win._on_source_changed(index)
            check("a hand-set value survives a source change",
                  win.current_settings().gain_db == by_hand,
                  f"{by_hand} -> {win.current_settings().gain_db}")

        win.reset_defaults()
        check("Reset restores the dialog defaults",
              win.current_settings().color_scheme == Settings().color_scheme)
    finally:
        win.close()
        app.processEvents()


def test_real_audacity_exports() -> None:
    """Ground truth: eight exports of one clip, four schemes x two axes.

    Synthetic images are drawn using the same tables they are then read back
    with, so they can only catch a coding mistake, never a wrong table.  These
    eight files are the only place the tables face colours this project did not
    produce, which is why the derivation is checked here.
    """
    print("\nReal Audacity exports (ground truth from filenames)")
    from spectro import colormap

    root = Path(__file__).resolve().parent.parent / "testing-files"
    schemes = {
        1: "Audacity Color (default)",
        2: "Audacity Color (classic)",
        3: "Audacity Grayscale",
        4: "Audacity Inverse Grayscale",
    }
    missing = [f"{ax}{'' if i == 1 else i}.png" for ax in ("linear", "mel")
               for i in schemes
               if not (root / f"{ax}{'' if i == 1 else i}.png").is_file()]
    if missing:
        print(f"  SKIP - reference exports not present ({len(missing)} missing)")
        return

    worst_own = 0.0
    for axis in ("linear", "mel"):
        for index, name in schemes.items():
            path = root / f"{axis}{'' if index == 1 else index}.png"
            rgb = core.load_image(str(path))

            own = colormap.fit_error(rgb, name)
            worst_own = max(worst_own, own)
            check(f"{path.name}: its own table fits", own < 2.0,
                  f"{own:.2f}/255")

    check("every table fits its own export within 2/255", worst_own < 2.0,
          f"worst {worst_own:.2f}/255")

    # The tables must not be interchangeable, or "reading it through the scheme
    # you picked" would not mean anything.
    rgb = core.load_image(str(root / "linear.png"))
    own = colormap.fit_error(rgb, "Audacity Color (default)")
    other = colormap.fit_error(rgb, "Audacity Color (classic)")
    check("a wrong table fits far worse than the right one", other > 10 * own,
          f"own {own:.2f} vs other {other:.2f}")

    # A colour table has nothing to do with the frequency axis, so the same
    # scheme must read equally well off a linear and a mel export.
    for index, name in schemes.items():
        lin = colormap.fit_error(core.load_image(
            str(root / f"{'linear' if index == 1 else f'linear{index}'}.png")), name)
        mel = colormap.fit_error(core.load_image(
            str(root / f"{'mel' if index == 1 else f'mel{index}'}.png")), name)
        check(f"{name}: reads the same off either axis", abs(lin - mel) < 1.0,
              f"linear {lin:.2f} vs mel {mel:.2f}")


def main() -> int:
    tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/spectro-tests")
    test_level_mapping()
    test_scales()
    test_windows_and_stft()
    test_time_control()
    test_no_edge_burst()
    test_edge_cases()
    test_image_formats(tmp)
    test_colour_schemes()
    test_table_fit()
    test_source_catalogue()
    test_tool_palettes()
    test_denoise()
    test_orientation()
    test_preview_region()
    test_preview_region_shortens_the_audio()
    test_pghi()
    test_pghi_gamma()
    test_window_should_match_the_export()
    test_audacity_tables()
    test_real_audacity_exports()
    test_window_does_not_grow(tmp / 'gui')
    test_zoom_and_pan(tmp / 'gui')
    test_preview_replaces_playback()
    test_calibration_names_the_right_ends()
    test_crop_preview()
    test_noise_profile()
    test_pitch()
    test_crop_limit()
    test_image_controls_wait_for_an_image()
    test_readouts_agree_about_dbfs()
    test_ffmpeg_sources_are_calibrated()
    test_text_is_readable()
    test_bin_ratio_advice()
    test_gui_source_selection()
    test_roundtrip()
    print("\n" + ("ALL PASSED" if not FAILURES else f"FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
