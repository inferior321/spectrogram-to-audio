"""Writing audio out, and playing it back.

MP3 is written by libsndfile when the installed build supports it (1.1+), and
by piping to ffmpeg otherwise.  The venv here has libsndfile 1.2.2, so the
first path is normally taken and ffmpeg is only a safety net.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

_SUBTYPES = {
    "16-bit int": "PCM_16",
    "24-bit int": "PCM_24",
    "32-bit float": "FLOAT",
}


def _libsndfile_has_mp3() -> bool:
    try:
        return "MP3" in sf.available_formats()
    except Exception:
        return False


def export(audio: np.ndarray, sample_rate: int, path: str | Path,
           fmt: str = "mp3", bitrate_kbps: int = 320,
           bit_depth: str = "16-bit int") -> Path:
    """Write `audio` (mono float32 in -1..1) and return the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower().lstrip(".")

    if fmt == "wav":
        sf.write(path, audio, sample_rate, subtype=_SUBTYPES.get(bit_depth, "PCM_16"))
    elif fmt == "flac":
        # FLAC is integer-only; 32-bit float would be silently refused.
        sub = _SUBTYPES.get(bit_depth, "PCM_16")
        sf.write(path, audio, sample_rate,
                 subtype=sub if sub in ("PCM_16", "PCM_24") else "PCM_24")
    elif fmt == "ogg":
        sf.write(path, audio, sample_rate, format="OGG", subtype="VORBIS")
    elif fmt == "mp3":
        if _libsndfile_has_mp3():
            sf.write(path, audio, sample_rate, format="MP3")
        else:
            _ffmpeg_mp3(audio, sample_rate, path, bitrate_kbps)
    else:
        raise ValueError(f"unsupported output format: {fmt!r}")
    return path


def _ffmpeg_mp3(audio: np.ndarray, sample_rate: int, path: Path, bitrate_kbps: int) -> None:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "MP3 export needs either a libsndfile build with MP3 support or "
            "ffmpeg on PATH; neither was found. Export to WAV or OGG instead."
        )
    cmd = [
        exe, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-b:a", f"{int(bitrate_kbps)}k", str(path),
    ]
    proc = subprocess.run(cmd, input=audio.astype(np.float32).tobytes(),
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:400]}")


# --------------------------------------------------------------------------
# Playback
# --------------------------------------------------------------------------

class Player:
    """Thin wrapper over sounddevice that degrades to silence, not a crash.

    A machine with no audio device (or no PortAudio) should still be able to
    convert and export files, so every failure here is reported rather than
    raised at import time.
    """

    def __init__(self) -> None:
        self._sd = None
        self.error: str | None = None
        try:
            import sounddevice as sd
            self._sd = sd
        except Exception as exc:                # pragma: no cover - host specific
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._sd is not None

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        if not self._sd:
            raise RuntimeError(self.error or "no audio backend")
        self._sd.stop()
        self._sd.play(audio, sample_rate)

    def stop(self) -> None:
        if self._sd:
            self._sd.stop()

    def is_playing(self) -> bool:
        if not self._sd:
            return False
        try:
            return bool(self._sd.get_stream().active)
        except Exception:
            return False
