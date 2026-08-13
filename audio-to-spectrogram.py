#!/usr/bin/env python3
"""
Spectrogrammer - a small Tk GUI around ffmpeg's showspectrumpic filter.

Companion to spectrogram-to-audio: this makes the pictures, that reads them
back. Nothing here is imported by the main program and the main program is not
imported here; they only agree about what the images mean.

Needs nothing from pip - only the standard library - but two things must be on
the system:

    ffmpeg      does all the work. Without it the Generate button refuses.
    python3-tk  the GUI toolkit. Not installable with pip; on Debian and
                Ubuntu it is a separate package from python itself.

        sudo apt install ffmpeg python3-tk

ffprobe (shipped with ffmpeg) is used for the duration and sample-rate
readouts and is required for "Split into chunks" to do anything.

Run:
    python3 audio-to-spectrogram.py
"""

import os
import queue
import shlex
import shutil
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# ---------------------------------------------------------------- options ---

# showspectrumpic has no FFT-size knob: the transform length is exactly twice
# the FREQUENCY-axis dimension in pixels (no power-of-two rounding). Which
# dimension that is depends on orientation, measured on ffmpeg 6.1.1:
#     orientation=horizontal -> frequency runs along the WIDTH,  time along the height
#     orientation=vertical   -> frequency runs along the HEIGHT, time along the width
# The live "Analysis" readout below the Size box shows the resulting FFT size,
# so it stays correct when you flip orientation.
# Ordered by what they are good for, because the shape matters more than the
# size. Resynthesis wants MANY time columns and only a moderate frequency axis:
# a picture with too few columns per second is stretched over the frames when it
# is read back, and the missing detail gets invented. It also helps if the
# frequency axis is a power of two, since the FFT is exactly twice it and a
# reader whose transform sizes are powers of two can then match it exactly
# rather than resampling onto the nearest grid.
SIZES = [
    # Wide: made for reading back into audio. FFT 4096, so a 4096-sample
    # window with no zero padding matches these exactly.
    "16384x2048", "32768x2048", "65536x2048",
    # Square-ish: finer frequency detail, fewer columns per second.
    "1024x512", "4096x2048", "4096x4096", "8192x4096",
    "8192x8192", "16384x8192", "8192x16384", "16384x16384",
    # Screen shaped: for looking at. The frequency axis is not a power of two,
    # so the FFT lands off-grid (1080 px -> 2160) and a reader has to resample.
    "800x600", "1280x720", "1920x1080", "2560x1440", "3840x2160",
]

COLORS = [
    "intensity", "rainbow", "moreland", "nebulae", "fire", "fiery",
    "fruit", "cool", "magma", "green", "viridis", "plasma",
    "cividis", "terrain", "channel",
]

# amplitude mapping
SCALES = ["log", "lin", "sqrt", "cbrt", "4thrt", "5thrt"]

# frequency axis
FSCALES = ["lin", "log"]

WIN_FUNCS = [
    "hann", "hamming", "blackman", "bartlett", "welch", "flattop",
    "bharris", "bnuttall", "bhann", "sine", "nuttall", "lanczos",
    "gauss", "tukey", "dolph", "cauchy", "parzen", "poisson",
    "bohman", "rect",
]

MODES = ["combined", "separate"]

ORIENTATIONS = ["horizontal", "vertical"]

FORMATS = ["png", "jpg", "webp", "tiff", "bmp"]

# formats that choke on the filter's RGBA output and need a conversion
NEEDS_RGB24 = {"jpg", "bmp"}

AUDIO_TYPES = [
    ("Audio & video", "*.flac *.wav *.mp3 *.m4a *.aac *.ogg *.opus *.wv "
                      "*.ape *.alac *.aiff *.aif *.mka *.dsf *.wma "
                      "*.mkv *.mp4 *.m4v *.avi *.mov *.webm *.flv *.wmv "
                      "*.mpg *.mpeg *.ts *.m2ts *.vob *.ogv *.3gp"),
    ("Audio only", "*.flac *.wav *.mp3 *.m4a *.aac *.ogg *.opus *.wv "
                   "*.ape *.alac *.aiff *.aif *.mka *.dsf *.wma"),
    ("Video only", "*.mkv *.mp4 *.m4v *.avi *.mov *.webm *.flv *.wmv "
                   "*.mpg *.mpeg *.ts *.m2ts *.vob *.ogv *.3gp"),
    ("All files", "*.*"),
]


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.files = []
        self.msgq = queue.Queue()
        self.running = False
        # ffprobe results, keyed by (path, track). Every control change rebuilds
        # the readouts below, and probing on each one meant launching two
        # subprocesses per keystroke - enough to stall the window on a large
        # file. The track number is part of the key, so switching tracks
        # re-probes without needing to clear anything.
        self._probes = {}

        self._build_vars()
        self._build_ui()
        self._poll_queue()
        self._refresh_cmd()

    # ------------------------------------------------------------- state ---

    def _build_vars(self):
        self.v_outdir = tk.StringVar(value="")
        self.v_size = tk.StringVar(value="4096x2048")
        self.v_color = tk.StringVar(value="intensity")
        self.v_scale = tk.StringVar(value="log")
        self.v_fscale = tk.StringVar(value="lin")
        self.v_win = tk.StringVar(value="hann")
        self.v_mode = tk.StringVar(value="combined")
        # vertical is ffmpeg's own default and the only one that puts time
        # along the width. horizontal transposes the picture, which is fine to
        # look at and useless to read back unless you know to rotate it.
        self.v_orient = tk.StringVar(value="vertical")
        self.v_fmt = tk.StringVar(value="png")
        self.v_gain = tk.DoubleVar(value=1.0)
        self.v_sat = tk.DoubleVar(value=1.0)
        self.v_drange = tk.DoubleVar(value=120.0)
        self.v_legend = tk.BooleanVar(value=False)
        self.v_mono = tk.BooleanVar(value=True)
        self.v_track = tk.IntVar(value=0)
        self.v_chunk = tk.BooleanVar(value=False)
        self.v_chunklen = tk.DoubleVar(value=30.0)
        self.v_status = tk.StringVar(value="Ready.")

        for var in (self.v_size, self.v_color, self.v_scale, self.v_fscale,
                    self.v_win, self.v_mode, self.v_orient, self.v_fmt,
                    self.v_gain, self.v_sat, self.v_drange, self.v_legend,
                    self.v_mono, self.v_chunk, self.v_chunklen, self.v_track,
                    self.v_outdir):
            var.trace_add("write", lambda *_: self._refresh_cmd())

    # ---------------------------------------------------------------- ui ---

    def _build_ui(self):
        row = 0

        # --- input files ---
        ttk.Label(self, text="Input files").grid(row=row, column=0, sticky="nw")
        box = ttk.Frame(self)
        box.grid(row=row, column=1, columnspan=3, sticky="ew", pady=(0, 8))
        box.columnconfigure(0, weight=1)

        self.lst = tk.Listbox(box, height=5, selectmode=tk.EXTENDED)
        self.lst.grid(row=0, column=0, sticky="ew")
        sb = ttk.Scrollbar(box, orient="vertical", command=self.lst.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.lst.config(yscrollcommand=sb.set)

        btns = ttk.Frame(box)
        btns.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(btns, text="Add...", command=self.add_files).pack(side="left")
        ttk.Button(btns, text="Remove", command=self.remove_selected).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear", command=self.clear_files).pack(side="left")
        row += 1

        # --- output dir ---
        ttk.Label(self, text="Output folder").grid(row=row, column=0, sticky="w")
        outrow = ttk.Frame(self)
        outrow.grid(row=row, column=1, columnspan=3, sticky="ew", pady=2)
        outrow.columnconfigure(0, weight=1)
        ttk.Entry(outrow, textvariable=self.v_outdir).grid(row=0, column=0, sticky="ew")
        ttk.Button(outrow, text="Browse...", command=self.pick_outdir).grid(row=0, column=1, padx=(4, 0))
        row += 1
        ttk.Label(self, text="(blank = alongside each input file)",
                  foreground="grey").grid(row=row, column=1, sticky="w", pady=(0, 8))
        row += 1

        # --- settings grid ---
        opts = ttk.LabelFrame(self, text="Spectrogram settings", padding=8)
        opts.grid(row=row, column=0, columnspan=4, sticky="ew", pady=4)
        for c in (1, 3):
            opts.columnconfigure(c, weight=1)

        def combo(parent, r, c, label, var, values, width=14):
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky="e", padx=(0, 6), pady=3)
            w = ttk.Combobox(parent, textvariable=var, values=values,
                             width=width, state="readonly")
            w.grid(row=r, column=c + 1, sticky="w", pady=3)
            return w

        # size is editable so custom WxH is allowed
        ttk.Label(opts, text="Size").grid(row=0, column=0, sticky="e", padx=(0, 6), pady=3)
        ttk.Combobox(opts, textvariable=self.v_size, values=SIZES,
                     width=18).grid(row=0, column=1, sticky="w", pady=3)

        self.v_fft = tk.StringVar(value="")
        ttk.Label(opts, textvariable=self.v_fft, foreground="grey").grid(
            row=6, column=0, columnspan=4, sticky="w", pady=(6, 0))

        combo(opts, 0, 2, "Colour map", self.v_color, COLORS)
        combo(opts, 1, 0, "Amplitude", self.v_scale, SCALES)
        combo(opts, 1, 2, "Freq. axis", self.v_fscale, FSCALES)
        combo(opts, 2, 0, "Window", self.v_win, WIN_FUNCS)
        combo(opts, 2, 2, "Channels", self.v_mode, MODES)
        combo(opts, 3, 0, "Orientation", self.v_orient, ORIENTATIONS)
        combo(opts, 3, 2, "Format", self.v_fmt, FORMATS)

        def spin(parent, r, c, label, var, lo, hi, step):
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky="e", padx=(0, 6), pady=3)
            ttk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=step,
                        width=12).grid(row=r, column=c + 1, sticky="w", pady=3)

        spin(opts, 4, 0, "Gain", self.v_gain, 0.1, 10.0, 0.1)
        spin(opts, 4, 2, "Saturation", self.v_sat, -10.0, 10.0, 0.1)
        spin(opts, 5, 0, "Dyn. range (dB)", self.v_drange, 20.0, 200.0, 10.0)
        ttk.Checkbutton(opts, text="Draw legend / axes",
                        variable=self.v_legend).grid(row=5, column=2, columnspan=2,
                                                     sticky="w", padx=(0, 6))
        row += 1

        # --- resynthesis helpers ---
        res = ttk.LabelFrame(self, text="For resynthesis", padding=8)
        res.grid(row=row, column=0, columnspan=4, sticky="ew", pady=4)
        res.columnconfigure(3, weight=1)

        ttk.Checkbutton(res, text="Force mono (-ac 1)",
                        variable=self.v_mono).grid(row=0, column=0, columnspan=2,
                                                   sticky="w", pady=2)
        ttk.Label(res, text="Audio track:").grid(row=0, column=2, sticky="e", padx=(12, 4))
        ttk.Spinbox(res, textvariable=self.v_track, from_=0, to=32,
                    increment=1, width=6).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(res, text="Split into chunks of",
                        variable=self.v_chunk).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(res, textvariable=self.v_chunklen, from_=1.0, to=600.0,
                    increment=5.0, width=8).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(res, text="seconds").grid(row=1, column=2, sticky="w")

        self.v_density = tk.StringVar(value="")
        ttk.Label(res, textvariable=self.v_density, foreground="grey").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        row += 1

        # --- command preview ---
        ttk.Label(self, text="Command").grid(row=row, column=0, sticky="nw", pady=(8, 0))
        self.cmd_box = tk.Text(self, height=4, wrap="word")
        self.cmd_box.grid(row=row, column=1, columnspan=3, sticky="ew", pady=(8, 4))
        self.cmd_box.config(state="disabled")
        row += 1

        # --- actions ---
        act = ttk.Frame(self)
        act.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        act.columnconfigure(2, weight=1)
        self.btn_run = ttk.Button(act, text="Generate", command=self.run)
        self.btn_run.grid(row=0, column=0)
        ttk.Button(act, text="Copy command",
                   command=self.copy_cmd).grid(row=0, column=1, padx=6)
        self.prog = ttk.Progressbar(act, mode="determinate")
        self.prog.grid(row=0, column=2, sticky="ew", padx=6)
        row += 1

        ttk.Label(self, textvariable=self.v_status).grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------ actions ---

    def add_files(self):
        self._probes.clear()
        paths = filedialog.askopenfilenames(title="Select audio files",
                                            filetypes=AUDIO_TYPES)
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.lst.insert(tk.END, os.path.basename(p))
        self._refresh_cmd()

    def remove_selected(self):
        self._probes.clear()
        for i in reversed(self.lst.curselection()):
            self.lst.delete(i)
            del self.files[i]
        self._refresh_cmd()

    def clear_files(self):
        self._probes.clear()
        self.lst.delete(0, tk.END)
        self.files.clear()
        self._refresh_cmd()

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Output folder")
        if d:
            self.v_outdir.set(d)

    def copy_cmd(self):
        self.clipboard_clear()
        self.clipboard_append(self.cmd_box.get("1.0", tk.END).strip())
        self.v_status.set("Command copied to clipboard.")

    # ----------------------------------------------------------- ffmpeg ----

    def _size_value(self):
        """'8192x8192 (FFT 16384)' -> '8192x8192'. Bare 'WxH' passes through."""
        return self.v_size.get().split("(")[0].strip()

    def _axes(self):
        """(freq_axis_px, time_axis_px) for the current size and orientation.

        Verified against ffmpeg 6.1.1 by measuring the temporal smear of a
        single-sample impulse: horizontal puts frequency on the width,
        vertical puts it on the height.
        """
        try:
            w, h = (int(v) for v in self._size_value().lower().split("x")[:2])
        except (ValueError, IndexError):
            return None, None
        if w <= 0 or h <= 0:
            return None, None
        return (w, h) if self.v_orient.get() == "horizontal" else (h, w)

    def _probe(self, src):
        """(duration_seconds, sample_rate) for the selected stream, cached.

        One ffprobe call gets both. Anything missing comes back as None and
        every caller is expected to cope, since ffprobe may be absent entirely.
        """
        try:
            track = max(int(self.v_track.get()), 0)
        except (tk.TclError, ValueError):
            track = 0
        key = (str(src), track)
        if key in self._probes:
            return self._probes[key]

        result = (None, None)
        if shutil.which("ffprobe"):
            cmd = ["ffprobe", "-v", "error", "-select_streams", f"a:{track}",
                   "-show_entries", "stream=duration,sample_rate",
                   "-show_entries", "format=duration",
                   "-of", "default=nw=1:nk=1", str(src)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                dur = rate = None
                for line in r.stdout.split():
                    try:
                        val = float(line)
                    except ValueError:
                        continue          # ffprobe prints N/A for absent fields
                    # A sample rate is a whole number in the audible-device
                    # range; a duration is anything else. Telling them apart
                    # this way avoids depending on the order ffprobe prints in.
                    if val.is_integer() and 4000 <= val <= 768000 and rate is None:
                        rate = int(val)
                    elif val > 0 and dur is None:
                        dur = val
                result = (dur, rate)
            except (OSError, subprocess.TimeoutExpired):
                result = (None, None)

        self._probes[key] = result
        return result

    def _samplerate(self, src):
        """Sample rate of the selected audio stream, or None."""
        return self._probe(src)[1]

    def _update_fft(self):
        """Show the FFT length implied by the current size + orientation."""
        fax, _ = self._axes()
        if not fax:
            self.v_fft.set("Size must look like  WIDTHxHEIGHT.")
            return
        n = 2 * fax
        horizontal = self.v_orient.get() == "horizontal"
        axis = "width" if horizontal else "height"
        msg = f"FFT size {n}  (freq axis = {axis}, {fax} px)"
        sr = self._samplerate(self.files[0]) if self.files else None
        if sr:
            msg += f"   window {1000.0 * n / sr:.0f} ms,  {sr / n:.2f} Hz per bin"
        if horizontal:
            # Worth saying plainly: a transposed picture read as a normal one
            # turns frequency into time, so it does not sound slightly wrong,
            # it sounds like nothing.
            msg += ("\nTRANSPOSED: frequency runs along the width and time down "
                    "the height.\nFine to look at; to read it back, rotate it "
                    "first or use vertical.")
        elif fax & (fax - 1):
            # Not a power of two, so the FFT is not either, and a reader whose
            # transform sizes are powers of two has to resample the rows.
            msg += (f"\nFFT {n} is not a power of two, so a reader will resample "
                    f"onto the nearest grid.\nA frequency axis of "
                    f"{1 << (fax.bit_length() - 1)} or {1 << fax.bit_length()} px "
                    "would match exactly.")
        self.v_fft.set(msg)

    def _filter_string(self):
        fmt = self.v_fmt.get()
        parts = [
            f"s={self._size_value()}",
            f"mode={self.v_mode.get()}",
            f"color={self.v_color.get()}",
            f"scale={self.v_scale.get()}",
            f"fscale={self.v_fscale.get()}",
            f"win_func={self.v_win.get()}",
            f"orientation={self.v_orient.get()}",
            f"gain={self.v_gain.get():g}",
            f"saturation={self.v_sat.get():g}",
            f"drange={self.v_drange.get():g}",
            f"legend={1 if self.v_legend.get() else 0}",
        ]
        flt = "showspectrumpic=" + ":".join(parts)
        if fmt in NEEDS_RGB24:
            flt += ",format=rgb24"
        try:
            track = max(int(self.v_track.get()), 0)
        except (tk.TclError, ValueError):
            track = 0
        return f"[0:a:{track}]" + flt

    def _out_path(self, src, index=None):
        src = Path(src)
        outdir = Path(self.v_outdir.get()) if self.v_outdir.get() else src.parent
        tag = "" if index is None else f"_{index:03d}"
        return outdir / f"{src.stem}_spectrogram{tag}.{self.v_fmt.get()}"

    def _build_cmd(self, src, start=None, dur=None, index=None):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if start is not None:
            cmd += ["-ss", f"{start:g}"]
        cmd += ["-i", str(src)]
        if dur is not None:
            cmd += ["-t", f"{dur:g}"]
        if self.v_mono.get():
            cmd += ["-ac", "1"]
        cmd += ["-lavfi", self._filter_string(), str(self._out_path(src, index))]
        return cmd

    def _duration(self, src):
        """Length of the selected audio stream in seconds, or None."""
        return self._probe(src)[0]

    def _jobs_for(self, src):
        """[(start, dur, index)] - one entry per image to render for this file."""
        if not self.v_chunk.get():
            return [(None, None, None)]
        length = self._duration(src)
        step = max(self.v_chunklen.get(), 0.1)
        if not length:
            return [(None, None, None)]
        n = max(1, int(length / step) + (1 if length % step > 0.05 else 0))
        return [(i * step, step, i) for i in range(n)]

    def _update_density(self):
        """Warn when columns-per-second is too low to resynthesise cleanly."""
        _, width = self._axes()   # columns run along the TIME axis
        if not width:
            self.v_density.set("")
            return
        if self.v_chunk.get() and not shutil.which("ffprobe"):
            # Without ffprobe the length is unknown, so _jobs_for falls back to
            # a single job and chunking quietly does nothing at all. Say so,
            # rather than letting one image come out where several were asked
            # for.
            self.v_density.set("ffprobe not found - chunking will be IGNORED "
                               "and one image written per file.")
            return
        span = self.v_chunklen.get() if self.v_chunk.get() else None
        if span is None and self.files:
            span = self._duration(self.files[0])
        if not span:
            self.v_density.set("Add a file to see time resolution.")
            return
        cps = width / span
        hop = 1000.0 / cps
        verdict = ("too coarse - will sound choppy" if cps < 40 else
                   "usable" if cps < 100 else "good")
        self.v_density.set(
            f"{cps:.0f} columns/sec  (~{hop:.0f} ms per column) - {verdict}")

    def _refresh_cmd(self, *_):
        sample = self.files[0] if self.files else "input.flac"
        jobs = [(None, None, None)] if not self.v_chunk.get() else [(0.0, self.v_chunklen.get(), 0)]
        cmd = " ".join(shlex.quote(a) for a in self._build_cmd(sample, *jobs[0]))
        self._update_density()
        self._update_fft()
        self.cmd_box.config(state="normal")
        self.cmd_box.delete("1.0", tk.END)
        self.cmd_box.insert("1.0", cmd)
        self.cmd_box.config(state="disabled")

    def run(self):
        if self.running:
            return
        if not shutil.which("ffmpeg"):
            messagebox.showerror("ffmpeg missing",
                                 "ffmpeg was not found on your PATH.\n\n"
                                 "Install it with:  sudo apt install ffmpeg")
            return
        if not self.files:
            messagebox.showwarning("No input", "Add at least one audio file first.")
            return

        outdir = self.v_outdir.get()
        if outdir:
            try:
                Path(outdir).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                messagebox.showerror("Bad output folder", str(e))
                return

        self.running = True
        self.btn_run.config(state="disabled")
        self.prog.config(maximum=len(self.files), value=0)
        threading.Thread(target=self._worker, args=(list(self.files),),
                         daemon=True).start()

    def _worker(self, files):
        ok = fail = 0
        errors = []

        jobs = []
        for src in files:
            for start, dur, index in self._jobs_for(src):
                jobs.append((src, start, dur, index))
        self.msgq.put(("total", len(jobs)))

        for n, (src, start, dur, index) in enumerate(jobs, 1):
            name = os.path.basename(src)
            label = name if index is None else f"{name} [{index:03d}]"
            self.msgq.put(("status", f"[{n}/{len(jobs)}] {label}"))
            try:
                r = subprocess.run(self._build_cmd(src, start, dur, index),
                                   capture_output=True, text=True, timeout=1800)
                if r.returncode == 0:
                    ok += 1
                else:
                    fail += 1
                    msg = (r.stderr or "").strip().splitlines()
                    errors.append(f"{label}: {msg[-1] if msg else 'unknown error'}")
            except subprocess.TimeoutExpired:
                fail += 1
                errors.append(f"{label}: timed out")
            except OSError as e:
                fail += 1
                errors.append(f"{label}: {e}")
            self.msgq.put(("progress", n))
        self.msgq.put(("done", (ok, fail, errors)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msgq.get_nowait()
                if kind == "status":
                    self.v_status.set(payload)
                elif kind == "total":
                    self.prog.config(maximum=max(payload, 1), value=0)
                elif kind == "progress":
                    self.prog.config(value=payload)
                elif kind == "done":
                    ok, fail, errors = payload
                    self.running = False
                    self.btn_run.config(state="normal")
                    self.v_status.set(f"Finished: {ok} written, {fail} failed.")
                    if errors:
                        messagebox.showerror(
                            "Some files failed",
                            "\n".join(errors[:12]) +
                            ("\n..." if len(errors) > 12 else ""))
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)


def main():
    root = tk.Tk()
    root.title("Spectrogrammer")
    root.minsize(720, 640)
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
