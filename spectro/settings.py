"""Settings model.

Every tunable lives here in one dataclass so that the GUI, the CLI and the
conversion core all agree on names, units and defaults.

The DEFAULTS are exactly the Audacity "Spectrogram Settings" dialog values the
project was started from (Linear scale, 1 Hz - 20000 Hz, Gain 20 dB,
Range 80 dB, High boost 0, Grayscale, Frequencies algorithm, window 2048, Hann,
zero padding factor 2).  Pressing "Reset" in the GUI restores precisely these.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from . import colormap

# --------------------------------------------------------------------------
# Enumerated choices (kept as plain strings so settings stay readable)
# --------------------------------------------------------------------------

FREQ_SCALES = ["Linear", "Logarithmic", "Mel", "Bark", "ERB", "Period"]

WINDOW_TYPES = [
    "Hann",
    "Hamming",
    "Blackman",
    "Blackman-Harris",
    "Bartlett",
    "Rectangular",
    "Welch",
    "Gaussian(a=3.5)",
    "Gaussian(a=4.5)",
]

# How pixel brightness becomes a level.
LEVEL_MAPPINGS = [
    "Audacity dB (gain/range)",  # matches the dialog exactly - see core.py
    "Plain dB range",            # white..black spans -range..0 dBFS
    "Linear amplitude",          # brightness IS the amplitude, no dB curve
    "Linear power",              # brightness is power, amplitude = sqrt
]

# How the picture is laid out.  Most tools put time along X and frequency up Y,
# but several - ffmpeg's showspectrum among them - can draw it rotated, and a
# rotated image read as if it were normal produces noise.  Both handednesses
# are offered rather than guessed at, because which way a tool rotates is not
# something to assume.
ORIENTATIONS = [
    "Time on X (normal)",
    "Time on Y, rotate clockwise",
    "Time on Y, rotate anticlockwise",
]

PHASE_INITS = ["PGHI (estimated)", "Random", "Zero", "Linear ramp"]

# Quality presets drive the phase settings together, since iterations, momentum
# and the starting phase only make sense as a combination.  The numbers come
# from measuring spectral convergence on real spectrogram images, not taste.
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    # The two fast presets start from noise on purpose.  PGHI is a Python walk
    # over every bin, so on a long clip it costs more than the whole draft
    # render does - 8.1s against 6.2s on a 4M-cell grid - and measured against
    # known audio it buys nothing on an image: 0.76 dB of error against Random's
    # 0.75.  Its real win is on true spectrogram magnitudes, whose gradients
    # have not been blurred by resampling and flattened to 8-bit.
    "Draft - fastest preview": dict(
        gl_iterations=16, gl_momentum=0.99, phase_init="Random"),
    "Fast": dict(
        gl_iterations=32, gl_momentum=0.99, phase_init="Random"),
    "Balanced (recommended)": dict(
        gl_iterations=100, gl_momentum=0.99, phase_init="PGHI (estimated)"),
    "High": dict(
        gl_iterations=300, gl_momentum=0.99, phase_init="PGHI (estimated)"),
    "Maximum (slow)": dict(
        gl_iterations=600, gl_momentum=0.99, phase_init="PGHI (estimated)"),
    "Classic Griffin-Lim": dict(
        gl_iterations=200, gl_momentum=0.0, phase_init="Random"),
    "Fast Griffin-Lim, random start": dict(
        gl_iterations=200, gl_momentum=0.99, phase_init="Random"),
    "Custom": {},
}

OUTPUT_FORMATS = ["mp3", "wav", "flac", "ogg"]

BIT_DEPTHS = ["16-bit int", "24-bit int", "32-bit float"]


@dataclass
class Settings:
    """All conversion parameters.  Units are stated in each comment."""

    # ---- Source image -----------------------------------------------------
    color_scheme: str = "Audacity Grayscale"
    # The scheme decides BOTH which colour means loud and how to read colour
    # pixels.  Audacity's Grayscale draws silence white and energy black, which
    # is what this project's own exports use.
    orientation: str = "Time on X (normal)"
    flip_polarity: bool = False      # override, when an image is upside-down in level
    crop_left: int = 0               # pixels trimmed before any processing
    crop_right: int = 0
    crop_top: int = 0
    crop_bottom: int = 0
    channel_mode: str = "Luminance"  # greyscale schemes only: Max/Red/Green/Blue

    # ---- Frequency axis ---------------------------------------------------
    freq_scale: str = "Linear"       # must be one of FREQ_SCALES
    min_freq: float = 1.0            # Hz, maps to the BOTTOM row of the image
    max_freq: float = 20000.0        # Hz, maps to the TOP row of the image

    # ---- Level mapping ----------------------------------------------------
    level_mapping: str = "Audacity dB (gain/range)"
    gain_db: float = 20.0            # Audacity "Gain (dB)"
    range_db: float = 80.0           # Audacity "Range (dB)"
    high_boost_db_per_dec: float = 0.0  # Audacity "High boost (dB/dec)", undone
    floor_gate: float = 0.0          # 0..1 level below which we force silence

    # ---- Analysis / FFT ---------------------------------------------------
    window_size: int = 2048          # Audacity "Window size"
    window_type: str = "Hann"        # Audacity "Window type"
    zero_padding: int = 2            # Audacity "Zero padding factor"
    overlap: int = 4                 # hop = window_size / overlap (resynthesis)

    # ---- Time axis --------------------------------------------------------
    sample_rate: int = 44100         # Hz
    duration_s: float = 0.0          # 0 or blank = infer from image width
    time_stretch: float = 1.0        # by-ear multiplier applied to the duration
    # Frequency multiplier applied when the image is read onto the FFT grid,
    # BEFORE phase reconstruction: 2.0 sounds an octave up, 0.5 an octave down.
    # Length is unaffected - only Duration, Time stretch and Sample rate move
    # that. 1.0 is off and costs nothing.
    pitch: float = 1.0

    # ---- Denoise ----------------------------------------------------------
    denoise_db: float = 0.0          # 0 = off; how hard to push the noise floor down
    denoise_percentile: float = 20.0 # per-bin floor estimated at this percentile

    # ---- Preview region ---------------------------------------------------
    # Fractions of the image width; 0..1 means the whole thing.
    preview_start: float = 0.0
    preview_end: float = 1.0

    # ---- Phase reconstruction --------------------------------------------
    quality: str = "Balanced (recommended)"
    gl_iterations: int = 100         # Fast Griffin-Lim iterations
    gl_momentum: float = 0.99        # 0 = classic Griffin-Lim, <=1 is stable
    phase_init: str = "PGHI (estimated)"
    random_seed: int = 0             # reproducible output

    # ---- Output -----------------------------------------------------------
    normalize: bool = True
    target_dbfs: float = -1.0        # peak target when normalize is on
    fade_ms: float = 10.0            # de-click fade at both ends
    output_format: str = "mp3"
    mp3_bitrate_kbps: int = 320
    bit_depth: str = "16-bit int"    # for wav/flac only

    # -- helpers ------------------------------------------------------------

    @property
    def n_fft(self) -> int:
        """FFT size actually used: window size times the zero padding factor."""
        return int(self.window_size * max(1, self.zero_padding))

    @property
    def hop(self) -> int:
        """Samples between successive frames on resynthesis."""
        return max(1, int(self.window_size // max(1, self.overlap)))

    def copy(self) -> "Settings":
        return Settings(**asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Tolerant loader - unknown keys are ignored, missing keys keep
        their default."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


DEFAULTS = Settings()
