"""The conversion engine: spectrogram image -> audio samples.

Pipeline
--------
    1. load + crop + flip the image so row 0 is the LOWEST frequency
    2. brightness -> level (0..1), inverting if the background is light
    3. level -> magnitude, using the same dB curve Audacity used to draw it
    4. resample the image grid onto the real (frequency bin x time frame) grid
    5. invent a phase with Fast Griffin-Lim
    6. inverse STFT -> samples

Only numpy/scipy/Pillow are used, so there is no numba/librosa weight and the
whole venv stays small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from scipy.signal import get_window

from . import colormap
from .settings import Settings

EPS = 1e-12
ProgressFn = Callable[[float, str], None]


class Cancelled(Exception):
    """Raised inside a progress callback to abort a long conversion."""


def _noop(frac: float, msg: str) -> None:
    return None


# ==========================================================================
# 1. Frequency-scale mathematics
# ==========================================================================
#
# Every supported scale is defined by a single monotonically increasing
# forward function s(f).  A pixel row y in 0..1 (bottom..top) then means
#
#       s(f) = s(fmin) + y * (s(fmax) - s(fmin))
#
# and we recover f by numerically inverting s on a dense grid.  Doing it this
# way means Bark and ERB - which have no closed-form inverse - cost no extra
# code than Linear does.

def _scale_forward(f: np.ndarray, scale: str) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    if scale == "Linear":
        return f
    if scale == "Logarithmic":
        return np.log(np.maximum(f, 1e-6))
    if scale == "Mel":                       # HTK mel, as used by Audacity
        return 2595.0 * np.log10(1.0 + f / 700.0)
    if scale == "Bark":                      # Traunmuller
        return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)
    if scale == "ERB":                       # Glasberg & Moore
        return 21.4 * np.log10(1.0 + 0.00437 * f)
    if scale == "Period":                    # 1/f, negated to stay increasing
        return -1.0 / np.maximum(f, 1e-6)
    raise ValueError(f"unknown frequency scale: {scale!r}")


def freq_to_unit(freqs: np.ndarray, fmin: float, fmax: float, scale: str) -> np.ndarray:
    """Map frequencies (Hz) to normalised image height 0..1 (0 = bottom row).

    Values outside the image range come back outside 0..1 on purpose - the
    caller uses that to zero out bins the picture never showed.
    """
    lo, hi = _scale_forward(np.array([fmin, fmax]), scale)
    if abs(hi - lo) < EPS:
        return np.zeros_like(np.asarray(freqs, dtype=np.float64))
    return (_scale_forward(freqs, scale) - lo) / (hi - lo)


# ==========================================================================
# 2. Image -> level image
# ==========================================================================

def load_image(path: str) -> np.ndarray:
    """Return the image as a float RGB array in 0..1, shape (H, W, 3)."""
    img = Image.open(path)
    if img.mode in ("P", "LA", "RGBA"):
        # Flatten transparency onto white, matching how such images are viewed.
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def apply_orientation(rgb: np.ndarray, st: Settings) -> np.ndarray:
    """Turn a rotated spectrogram the right way up: time along X, frequency up Y.

    Call this once, straight after loading.  Everything downstream - cropping,
    the time region, the frequency axis - then works in one orientation, and
    the picture on screen matches what is being read.

    A rotated image read as though it were normal is not subtly wrong, it is
    nonsense: the frequency axis becomes time and vice versa.
    """
    if st.orientation == "Time on Y, rotate clockwise":
        return np.ascontiguousarray(np.rot90(rgb, k=-1))
    if st.orientation == "Time on Y, rotate anticlockwise":
        return np.ascontiguousarray(np.rot90(rgb, k=1))
    return rgb


def crop_box(height: int, width: int, st: Settings) -> tuple[int, int, int, int]:
    """The crop actually applied, as (top, bottom, left, right).

    Clamped so at least one row and one column always survive: opposing crops
    that would meet in the middle - or exceed the image entirely - give an
    empty array, and every stage downstream would divide by its size. The
    display and the conversion both call this, so what you see cropped on
    screen is exactly the pixels that get turned into sound.
    """
    t = max(0, min(int(st.crop_top), height - 1))
    b = max(0, min(int(st.crop_bottom), height - 1 - t))
    l = max(0, min(int(st.crop_left), width - 1))
    r = max(0, min(int(st.crop_right), width - 1 - l))
    return t, b, l, r


def crop_view(rgb: np.ndarray, st: Settings) -> np.ndarray:
    """`rgb` with the crop applied. A view, not a copy."""
    h, w = rgb.shape[0], rgb.shape[1]
    t, b, l, r = crop_box(h, w, st)
    return rgb[t : h - b, l : w - r]


def to_level(rgb: np.ndarray, st: Settings) -> np.ndarray:
    """Collapse RGB to a single 0..1 'how loud was this pixel' image.

    Returns an array whose row 0 is the LOWEST frequency (image already
    flipped) and where 1.0 always means loud, regardless of the colour scheme.
    """
    view = crop_view(rgb, st)

    # Optional time window, used by the preview so a few seconds can be tried
    # without waiting for the whole image.
    vw = view.shape[1]
    if preview_fraction(st) < 1.0:
        a = int(round(np.clip(st.preview_start, 0.0, 1.0) * vw))
        z = int(round(np.clip(st.preview_end, 0.0, 1.0) * vw))
        # Always keep a few columns: a region can be narrower than this on a
        # small image, and cropping to nothing would be worse than rounding up.
        if z - a < 4:
            z = min(vw, a + 4)
            a = max(0, z - 4)
        view = view[:, a:z, :]

    scheme = st.color_scheme
    if scheme in colormap.BUILTIN:
        # Greyscale schemes let the channel choice matter; a colour map does
        # not, since it reads whole RGB triples.
        mode = st.channel_mode
        if mode == "Max channel":
            lev = view.max(axis=2)
        elif mode in ("Red", "Green", "Blue"):
            lev = view[:, :, "RGB".index(mode[0])]
        else:
            lev = view @ colormap.LUM
        if scheme == "Greyscale (light = quiet)":
            lev = 1.0 - lev
    else:
        lev = colormap.level_from_scheme(view, scheme)

    if st.flip_polarity:
        lev = 1.0 - lev

    lev = np.flipud(lev)                      # row 0 becomes the bottom = min_freq
    return np.clip(lev.astype(np.float32), 0.0, 1.0)


def level_to_db(level: float, st: Settings, *, silent_at_zero: bool = True) -> float:
    """Brightness 0..1 -> dBFS, by exactly the curve level_to_magnitude undoes.

    The hover readout and the calibration line both come through here. They
    used to each carry their own copy of the Audacity formula, so with any
    other mapping selected the two disagreed with each other and with the
    conversion - hovering black read -100 dBFS while the calibration line
    directly below it said the quiet end was -80.

    `silent_at_zero` is what the picture MEANS: an unlit pixel is silence, not
    the bottom of the scale. The calibration line passes False because it is
    naming where the ends of the scale sit, not reading a pixel.
    """
    lev = float(level)
    if silent_at_zero and lev <= max(0.0, st.floor_gate):
        return float("-inf")
    mapping = st.level_mapping
    if mapping == "Audacity dB (gain/range)":
        return lev * st.range_db - st.range_db - st.gain_db
    if mapping == "Plain dB range":
        return (lev - 1.0) * st.range_db
    if lev <= 0.0:
        return float("-inf")
    if mapping == "Linear power":
        return 10.0 * float(np.log10(lev))
    return 20.0 * float(np.log10(lev))         # "Linear amplitude"


def level_to_magnitude(level: np.ndarray, freqs: np.ndarray, st: Settings) -> np.ndarray:
    """Undo the drawing curve: 0..1 brightness -> linear STFT magnitude.

    `freqs` is the frequency of every row, needed only to undo High boost.
    """
    lev = level
    if st.floor_gate > 0:
        lev = np.where(lev <= st.floor_gate, 0.0, lev)

    mapping = st.level_mapping
    if mapping == "Audacity dB (gain/range)":
        # Audacity draws  level = (dB + range + gain) / range  clamped to 0..1,
        # where dB = 10*log10(power) = 20*log10(magnitude).  Inverting:
        #     dB = level*range - range - gain
        # With the dialog's Gain 20 / Range 80 that means white = -100 dBFS
        # and black = -20 dBFS, which is the real calibration of the picture.
        db = lev * st.range_db - st.range_db - st.gain_db
        mag = np.power(10.0, db / 20.0)
        mag = np.where(lev <= 0.0, 0.0, mag)   # true white is silence, not -100 dB
    elif mapping == "Plain dB range":
        db = (lev - 1.0) * st.range_db
        mag = np.power(10.0, db / 20.0)
        mag = np.where(lev <= 0.0, 0.0, mag)
    elif mapping == "Linear power":
        mag = np.sqrt(np.maximum(lev, 0.0))
    else:                                      # "Linear amplitude"
        mag = np.maximum(lev, 0.0)

    if abs(st.high_boost_db_per_dec) > 1e-9:
        # Audacity's High boost adds N dB per decade above the minimum
        # frequency when drawing; remove exactly that much to get back.
        f = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-6)
        f0 = max(st.min_freq, 1e-6)
        decades = np.log10(f / f0)
        boost_db = st.high_boost_db_per_dec * np.maximum(decades, 0.0)
        mag = mag / np.power(10.0, boost_db[:, None] / 20.0)

    return mag.astype(np.float32)


# ==========================================================================
# 3. Regridding image pixels -> STFT bins and frames
# ==========================================================================

def edge_trim(st: Settings) -> int:
    """Samples at each end that never receive full overlap-add coverage.

    Only after `overlap` frames have accumulated does every sample sit under a
    complete set of windows.  Before that the reconstruction concentrates into
    too few samples and produces a burst far louder than the real content, so
    these edges are cut off rather than kept.
    """
    return max(0, st.window_size - st.hop)


def raw_length(n_frames: int, st: Settings) -> int:
    """Samples produced by the inverse STFT, before the edges are trimmed."""
    return (n_frames - 1) * st.hop + st.window_size


def will_trim(n_frames: int, st: Settings) -> bool:
    """False when the clip is too short to give the edges away."""
    trim = edge_trim(st)
    return trim > 0 and raw_length(n_frames, st) - 2 * trim >= st.hop


def preview_fraction(st: Settings) -> float:
    """How much of the image's width the time region covers, 0..1.

    The duration you type describes the WHOLE image, so a region covering a
    twentieth of it is a twentieth of that duration.  Leaving this out meant a
    six per cent selection still synthesised the full length - 102 pixels of
    picture stretched over three and a half minutes - which made a preview
    slower than converting the whole file, and wrong as well.
    """
    lo = float(np.clip(st.preview_start, 0.0, 1.0))
    hi = float(np.clip(st.preview_end, 0.0, 1.0))
    frac = hi - lo
    # Matches the guard in to_level: a sliver that would not survive cropping
    # does not shorten the audio either.
    return frac if 0.0 < frac <= 1.0 else 1.0


def infer_frame_count(width: int, st: Settings) -> int:
    """Frames to synthesise.

    If the user gave a duration we honour it exactly, counting the extra frames
    needed to replace the edges that get trimmed away.  Otherwise we assume the
    export was one pixel column per FFT frame, which is what most tools do when
    you 'fit' a spectrogram to a window, and the time-stretch slider then lets
    the result be tuned by ear.
    """
    if st.duration_s and st.duration_s > 0:
        target = st.duration_s * preview_fraction(st) * st.time_stretch * st.sample_rate
        n = int(round((target + 2 * edge_trim(st) - st.window_size) / st.hop)) + 1
        # A duration short enough to keep no trimming needs the plain formula.
        if not will_trim(max(2, n), st):
            n = int(round((target - st.window_size) / st.hop)) + 1
    else:
        n = int(round(width * st.time_stretch))
    return max(2, n)


def frames_to_duration(n_frames: int, st: Settings) -> float:
    """Seconds of audio a given frame count produces, after trimming."""
    length = raw_length(n_frames, st)
    if will_trim(n_frames, st):
        length -= 2 * edge_trim(st)
    return length / float(st.sample_rate)


def _row_map(h: int, st: Settings):
    """Where each FFT bin reads from, in an image `h` rows tall.

    Returns the two neighbouring row indices, the blend weight between them, a
    mask of bins the image actually covers, and the SOURCE frequency of each
    bin. Shared so the noise profile lands on exactly the same grid as the
    picture it was cut from - a profile mapped even slightly differently would
    subtract the wrong bins.
    """
    bin_freqs = np.fft.rfftfreq(st.n_fft, d=1.0 / st.sample_rate)
    # Pitch: to make the output sound `pitch` times higher, the bin that will
    # be heard at frequency f must be filled from the part of the image drawn
    # at f / pitch.  Shifting here rather than on the finished audio means
    # phase reconstruction runs on the shifted spectrum and stays consistent
    # with it, so this adds no artifacts of its own and cannot change the
    # length - the frame count is decided by the time axis alone.
    pitch = float(st.pitch) if st.pitch and st.pitch > 0 else 1.0
    src_freqs = bin_freqs / pitch
    unit = freq_to_unit(src_freqs, st.min_freq, st.max_freq, st.freq_scale)
    rows = unit * (h - 1)
    # Bins whose source frequency falls off the image get nothing: shifting up
    # empties the top of the range, shifting down pushes content past Nyquist.
    inside = (unit >= 0.0) & (unit <= 1.0) & (bin_freqs <= st.sample_rate / 2.0)

    r0 = np.clip(np.floor(rows).astype(np.int64), 0, h - 1)
    r1 = np.clip(r0 + 1, 0, h - 1)
    wr = (rows - r0).astype(np.float32)
    return r0, r1, wr, inside, src_freqs


def noise_profile(rgb: np.ndarray, st: Settings) -> np.ndarray | None:
    """Average level per image row across the stretch marked as noise.

    Returns one number per row - a picture of what silence looks like in this
    image - or None when nothing is marked.  Computed from the WHOLE cropped
    image rather than from whatever the preview happens to be showing, so the
    noise patch does not have to be inside the region being previewed.
    """
    if not (float(st.noise_end) - float(st.noise_start) > 1e-6):
        return None
    probe = st.copy()
    probe.preview_start = float(st.noise_start)
    probe.preview_end = float(st.noise_end)
    level = to_level(rgb, probe)
    return level.mean(axis=1).astype(np.float32)


def build_magnitude(level: np.ndarray, st: Settings, progress: ProgressFn = _noop,
                    noise: np.ndarray | None = None) -> np.ndarray:
    """Resample the level image onto the (n_bins x n_frames) STFT grid.

    Interpolation happens in the *level* domain rather than on magnitudes:
    brightness is roughly logarithmic, so averaging there behaves like
    averaging decibels, which is much better behaved than averaging raw
    amplitudes across a 80 dB range.
    """
    h, w = level.shape
    n_fft = st.n_fft
    n_frames = infer_frame_count(w, st)

    # --- frequency axis: for every FFT bin, where does it sit in the image? --
    r0, r1, wr_flat, inside, src_freqs = _row_map(h, st)
    wr = wr_flat[:, None]
    progress(0.10, "Mapping frequency axis")

    # --- time axis: linear resample of columns ------------------------------
    cols = np.linspace(0.0, w - 1, n_frames)
    c0 = np.clip(np.floor(cols).astype(np.int64), 0, w - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    wc = (cols - c0).astype(np.float32)[None, :]

    # Bilinear in one shot: rows first (cheap, n_bins x w), then columns.
    stage = level[r0, :] * (1.0 - wr) + level[r1, :] * wr
    grid = stage[:, c0] * (1.0 - wc) + stage[:, c1] * wc
    progress(0.20, "Mapping time axis")

    grid[~inside, :] = 0.0                     # bins the image never showed
    # High boost is undone at the frequency the pixel was DRAWN at, which is
    # the source frequency - not the shifted one it will be heard at.
    mag = level_to_magnitude(grid, src_freqs, st)

    # Put the magnitudes on the same scale our own STFT produces.  A drawing
    # program reports 0 dB for a full-scale sine, whose STFT peak bin is
    # sum(window)/2 - without this factor the result is a constant ~54 dB
    # quieter than it claims, which normalisation hides but "peak dBFS" and
    # un-normalised exports do not.
    window = make_window(st.window_type, st.window_size)
    mag *= float(np.sum(window)) / 2.0

    if st.denoise_db > 0 and noise is not None and len(noise) == h:
        # Put the profile through the SAME row mapping and the same level
        # curve, so a noise row and a picture row of equal brightness become
        # equal magnitudes and the subtraction is like-for-like.
        col = noise[r0] * (1.0 - wr_flat) + noise[r1] * wr_flat
        col = np.where(inside, col, 0.0)[:, None]
        floor = level_to_magnitude(col, src_freqs, st)[:, 0]
        floor *= float(np.sum(window)) / 2.0
        mag = denoise(mag, st, floor)

    # A magnitude spectrum must be real and non-negative at DC and Nyquist.
    mag[0, :] = 0.0                            # kill DC so there is no offset
    return np.ascontiguousarray(mag, dtype=np.float32)


def denoise(mag: np.ndarray, st: Settings, floor: np.ndarray) -> np.ndarray:
    """Push a measured noise floor down without touching the signal above it.

    `floor` is one magnitude per frequency bin, taken from the stretch of image
    the user marked as noise. Each bin is attenuated by how far it sits above
    its own floor, which matters because a picture's background is not flat: a
    single global threshold eats quiet high frequencies while leaving
    low-frequency rumble untouched.

    This runs on the magnitudes before phase reconstruction, so Griffin-Lim
    never has to invent phase for noise that is about to be removed - which is
    also where its watery character comes from.

    The floor used to be guessed as a low percentile over time. That assumed
    the noise was steady and that most of any given bin was silence, which is
    wrong for music, and it is why the guessed version disappointed.
    """
    sens = max(0.05, float(st.denoise_sensitivity))
    threshold = np.maximum(np.asarray(floor, dtype=np.float32)[:, None] * sens, EPS)
    # Smoothly go from full attenuation at the threshold to untouched above it.
    gain = np.clip((mag / threshold - 1.0) / 2.0, 0.0, 1.0)

    # Frequency smoothing: average the gain over neighbouring bins. Subtraction
    # on its own leaves isolated bins flicking between kept and cut, which is
    # heard as "musical noise" - a warbling of tones that were never there.
    k = int(max(0, st.denoise_smoothing))
    if k > 0 and gain.shape[0] > 2 * k + 1:
        width = 2 * k + 1
        padded = np.pad(gain, ((k, k), (0, 0)), mode="edge")
        cums = np.concatenate(
            [np.zeros((1, gain.shape[1]), dtype=np.float32),
             np.cumsum(padded, axis=0, dtype=np.float32)], axis=0)
        rows = gain.shape[0]
        gain = (cums[width:width + rows] - cums[:rows]) / float(width)

    reduction = 10.0 ** (-abs(st.denoise_db) / 20.0)
    return (mag * (reduction + (1.0 - reduction) * gain)).astype(np.float32)


# ==========================================================================
# 4. STFT / ISTFT
# ==========================================================================

_WINDOW_ALIASES = {
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Blackman-Harris": "blackmanharris",
    "Bartlett": "bartlett",
    "Rectangular": "boxcar",
    "Welch": None,                             # scipy has no Welch window
    "Gaussian(a=3.5)": ("gaussian", 3.5),
    "Gaussian(a=4.5)": ("gaussian", 4.5),
}


def make_window(name: str, size: int) -> np.ndarray:
    spec = _WINDOW_ALIASES.get(name, "hann")
    if spec is None:                           # Welch, computed directly
        n = np.arange(size)
        return (1.0 - ((n - (size - 1) / 2.0) / ((size - 1) / 2.0)) ** 2).astype(np.float64)
    if isinstance(spec, tuple):
        base, alpha = spec
        # Audacity parametrises Gaussian by 'a'; sigma = size/(2a) matches it.
        return get_window((base, size / (2.0 * alpha)), size, fftbins=True)
    return get_window(spec, size, fftbins=True)


def stft(x: np.ndarray, st: Settings, window: np.ndarray) -> np.ndarray:
    """Frames x bins -> returned transposed as (bins, frames)."""
    win_len, hop, n_fft = st.window_size, st.hop, st.n_fft
    n_frames = 1 + max(0, (len(x) - win_len) // hop)
    idx = np.arange(win_len)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * window[None, :]
    if n_fft > win_len:                        # zero padding factor
        pad = np.zeros((n_frames, n_fft - win_len), dtype=frames.dtype)
        frames = np.concatenate([frames, pad], axis=1)
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def istft(spec: np.ndarray, st: Settings, window: np.ndarray, length: int) -> np.ndarray:
    """Weighted overlap-add inverse - the correct partner for Griffin-Lim."""
    win_len, hop = st.window_size, st.hop
    frames = np.fft.irfft(spec.T, n=st.n_fft, axis=1)[:, :win_len]
    n_frames = frames.shape[0]

    out = np.zeros(length, dtype=np.float64)
    wsum = np.zeros(length, dtype=np.float64)
    w2 = window ** 2
    for i in range(n_frames):
        a = i * hop
        b = min(a + win_len, length)
        if a >= length:
            break
        out[a:b] += frames[i, : b - a] * window[: b - a]
        wsum[a:b] += w2[: b - a]

    # The first and last half-window are covered by fewer frames than the
    # middle, and a tapered window is near zero at its own edges, so wsum
    # collapses towards zero there.  Dividing by that tiny number turns the
    # very first sample into an enormous spike - which then dominates peak
    # normalisation and, because Griffin-Lim feeds its output back in, is
    # amplified again on every iteration.  Clamping the denominator to a
    # fraction of its steady-state value leaves those incomplete edges quietly
    # attenuated instead, which is what they honestly are.
    steady = float(wsum.max()) if wsum.size else 1.0
    return out / np.maximum(wsum, max(1e-8, 0.05 * steady))


# ==========================================================================
# 5. Fast Griffin-Lim
# ==========================================================================

# Window time-frequency ratio constants: gamma = C * window_length^2 describes
# how fast a window's phase rotates, which is what turns log-magnitude
# gradients into phase gradients in pghi_phase below.
#
# For a Gaussian the constant is not a fitted number, it is exact.  This code
# builds a Gaussian with standard deviation sigma = length/(2a), and writing
# exp(-t^2 / 2 sigma^2) in the form exp(-pi t^2 / gamma) gives
# gamma = 2 pi sigma^2, so C = pi / (2 a^2).
#
# Guessing these instead of deriving them was a real mistake: a=3.5 was set to
# 0.245 where the truth is 0.128, and a=4.5 to 0.19 where the truth is 0.078.
# Being roughly twice too large made the phase estimate WORSE than starting
# from noise - on the very windows PGHI is derived for, and so works best with.
# The values for the other windows are the published LTFAT ones, and a sweep on
# true spectrogram magnitudes puts the optimum on top of them (Hann 0.256,
# Blackman 0.180), which is what says the formula around them is right.
def _gaussian_c(alpha: float) -> float:
    return float(np.pi / (2.0 * alpha ** 2))


_GAMMA_C = {
    "Hann": 0.25645,
    "Hamming": 0.29794,
    "Blackman": 0.17954,
    "Blackman-Harris": 0.13831,
    "Bartlett": 0.24000,
    "Rectangular": 0.85000,
    "Welch": 0.23000,
    "Gaussian(a=3.5)": _gaussian_c(3.5),
    "Gaussian(a=4.5)": _gaussian_c(4.5),
}


def pghi_phase(mag: np.ndarray, st: Settings, rel_tol: float = 1e-3,
               progress: ProgressFn = _noop) -> np.ndarray:
    """Estimate phase directly from the magnitudes, before any iteration.

    Phase Gradient Heap Integration.  For a Gaussian-like window the log
    magnitude and the phase of a spectrogram are not independent: the phase's
    rate of change in time is set by the log magnitude's slope in frequency,
    and vice versa.  So the phase can be integrated outwards across the
    time-frequency plane, and the integration is started from the loudest bin
    and always continues from the loudest bin reached so far - the heap - so
    that error accumulates through quiet regions rather than through the parts
    that matter.

    Bins below `rel_tol` of the peak are left with random phase; they are
    inaudible, and skipping them is what keeps this affordable, since the heap
    walk is a Python loop over every bin it visits.
    """
    n_bins, n_frames = mag.shape
    m_fft, hop = st.n_fft, st.hop
    gamma = _GAMMA_C.get(st.window_type, 0.25645) * (st.window_size ** 2)

    logs = np.log(np.maximum(mag.astype(np.float64), 1e-12))

    # Central differences, one-sided at the edges.
    dlog_dm = np.empty_like(logs)          # along frequency
    dlog_dm[1:-1, :] = (logs[2:, :] - logs[:-2, :]) / 2.0
    dlog_dm[0, :] = logs[1, :] - logs[0, :]
    dlog_dm[-1, :] = logs[-1, :] - logs[-2, :]

    dlog_dn = np.empty_like(logs)          # along time
    dlog_dn[:, 1:-1] = (logs[:, 2:] - logs[:, :-2]) / 2.0
    dlog_dn[:, 0] = logs[:, 1] - logs[:, 0]
    dlog_dn[:, -1] = logs[:, -1] - logs[:, -2]

    # Phase advance per frame and per frequency bin.
    #
    # Two terms come from this STFT's own conventions rather than from PGHI.
    # Frames are not demodulated, so a steady sinusoid in bin m rotates by
    # 2*pi*hop*m/M between frames.  And the window sits at the START of each
    # frame rather than centred on it, which multiplies the spectrum by a
    # linear phase ramp and shifts every frequency-direction step by a constant
    # -pi*window/M.  Leaving that ramp out makes the frequency gradient no
    # better than noise - at zero padding 2 it is wrong by exactly pi/2 per bin
    # - and the phase estimate ends up worse than starting from random.
    bins = np.arange(n_bins)[:, None]
    tgrad = (hop * m_fft / gamma) * dlog_dm + 2.0 * np.pi * hop * bins / m_fft
    fgrad = -(gamma / (hop * m_fft)) * dlog_dn - np.pi * st.window_size / m_fft

    rng = np.random.default_rng(st.random_seed)
    phase = rng.uniform(-np.pi, np.pi, mag.shape)
    done = np.zeros(mag.shape, dtype=bool)

    threshold = float(mag.max()) * rel_tol
    if threshold <= 0:
        return phase
    todo = mag > threshold
    if not todo.any():
        return phase

    # Plain Python lists: this loop indexes single elements millions of times,
    # and scalar indexing into a numpy array is several times more expensive.
    import heapq

    tg = tgrad.ravel().tolist()
    fg = fgrad.ravel().tolist()
    ph = phase.ravel().tolist()
    magf = mag.ravel()

    order = np.argsort(magf[todo.ravel()])[::-1]
    idx_todo = np.flatnonzero(todo.ravel())[order]
    heap: list[tuple[float, int]] = []
    seen = done.ravel()
    todo_flat = todo.ravel()
    total = len(idx_todo)
    processed = 0

    for start in idx_todo.tolist():
        if seen[start]:
            continue
        heapq.heappush(heap, (-magf[start], start))
        seen[start] = True
        while heap:
            _, k = heapq.heappop(heap)
            m, n = divmod(k, n_frames)
            processed += 1
            if processed % 262144 == 0:
                progress(0.03 + 0.16 * processed / max(1, total),
                         f"Estimating phase {100 * processed // max(1, total)}%")

            if n + 1 < n_frames:
                j = k + 1
                if todo_flat[j] and not seen[j]:
                    ph[j] = ph[k] + (tg[k] + tg[j]) / 2.0
                    seen[j] = True
                    heapq.heappush(heap, (-magf[j], j))
            if n > 0:
                j = k - 1
                if todo_flat[j] and not seen[j]:
                    ph[j] = ph[k] - (tg[k] + tg[j]) / 2.0
                    seen[j] = True
                    heapq.heappush(heap, (-magf[j], j))
            if m + 1 < n_bins:
                j = k + n_frames
                if todo_flat[j] and not seen[j]:
                    ph[j] = ph[k] + (fg[k] + fg[j]) / 2.0
                    seen[j] = True
                    heapq.heappush(heap, (-magf[j], j))
            if m > 0:
                j = k - n_frames
                if todo_flat[j] and not seen[j]:
                    ph[j] = ph[k] - (fg[k] + fg[j]) / 2.0
                    seen[j] = True
                    heapq.heappush(heap, (-magf[j], j))

    return np.asarray(ph, dtype=np.float64).reshape(mag.shape)


def griffin_lim(mag: np.ndarray, st: Settings, progress: ProgressFn = _noop) -> np.ndarray:
    """Recover a signal whose STFT magnitude matches `mag`.

    This is Griffin-Lim with the Perraudin momentum term ("Fast Griffin-Lim").
    momentum=0 gives classic Griffin-Lim; 0.99 converges roughly three times
    faster for the same error and is stable for any value <= 1.
    """
    window = make_window(st.window_type, st.window_size)
    n_frames = mag.shape[1]
    length = (n_frames - 1) * st.hop + st.window_size

    rng = np.random.default_rng(st.random_seed)
    if st.phase_init.startswith("PGHI"):
        progress(0.03, "Estimating phase (PGHI)")
        angles = np.exp(1j * pghi_phase(mag, st, progress=progress))
    elif st.phase_init == "Random":
        angles = np.exp(2j * np.pi * rng.random(mag.shape))
    elif st.phase_init == "Linear ramp":
        # A rising phase ramp per bin is a smoother starting point than noise
        # for tonal material and sometimes converges a little faster.
        k = np.arange(mag.shape[0])[:, None]
        t = np.arange(n_frames)[None, :]
        angles = np.exp(1j * 2 * np.pi * k * t * st.hop / st.n_fft)
    else:
        angles = np.ones(mag.shape, dtype=np.complex128)

    momentum = float(np.clip(st.gl_momentum, 0.0, 1.0))
    n_iter = max(0, int(st.gl_iterations))
    prev = np.zeros_like(angles)
    mag64 = mag.astype(np.float64)

    x = istft(mag64 * angles, st, window, length)
    for i in range(n_iter):
        rebuilt = stft(x, st, window)
        # Momentum: step past the plain projection using the previous estimate.
        c = rebuilt - (momentum / (1.0 + momentum)) * prev
        prev = rebuilt
        angles = c / np.maximum(np.abs(c), EPS)
        x = istft(mag64 * angles, st, window, length)
        if i % 4 == 0 or i == n_iter - 1:
            progress(0.25 + 0.70 * (i + 1) / max(1, n_iter),
                     f"Griffin-Lim iteration {i + 1}/{n_iter}")
    return x


# ==========================================================================
# 6. Top-level conversion
# ==========================================================================

@dataclass
class Result:
    audio: np.ndarray            # float32, mono, -1..1
    sample_rate: int
    duration_s: float
    n_frames: int
    magnitude: np.ndarray        # the reconstructed magnitude, for display
    peak_dbfs: float


def convert(level: np.ndarray, st: Settings, progress: ProgressFn = _noop,
            noise: np.ndarray | None = None) -> Result:
    """Run the whole pipeline on an already-loaded level image.

    `noise` is the per-row profile from core.noise_profile(), passed in
    rather than kept on Settings: it is an array derived from the picture,
    and Settings has to stay copyable and serialisable.
    """
    progress(0.02, "Building magnitude spectrogram")
    mag = build_magnitude(level, st, progress, noise)

    progress(0.25, "Reconstructing phase")
    x = griffin_lim(mag, st, progress)

    progress(0.96, "Finishing")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if will_trim(mag.shape[1], st):
        trim = edge_trim(st)
        x = x[trim:len(x) - trim]

    if st.fade_ms > 0:                         # remove the click at each end
        n = int(st.sample_rate * st.fade_ms / 1000.0)
        n = min(n, len(x) // 2)
        if n > 1:
            ramp = np.linspace(0.0, 1.0, n)
            x[:n] *= ramp
            x[-n:] *= ramp[::-1]

    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if st.normalize and peak > 0:
        x = x * (10.0 ** (st.target_dbfs / 20.0) / peak)
    elif peak > 1.0:                           # never clip on export
        x = x / peak
    peak_out = float(np.max(np.abs(x))) if len(x) else 0.0

    audio = np.clip(x, -1.0, 1.0).astype(np.float32)
    progress(1.0, "Done")
    return Result(
        audio=audio,
        sample_rate=st.sample_rate,
        duration_s=len(audio) / float(st.sample_rate),
        n_frames=mag.shape[1],
        magnitude=mag,
        peak_dbfs=20.0 * np.log10(peak_out) if peak_out > 0 else -np.inf,
    )


def convert_file(path: str, st: Settings, progress: ProgressFn = _noop) -> Result:
    rgb = load_image(path)
    return convert(to_level(rgb, st), st, progress, noise_profile(rgb, st))
