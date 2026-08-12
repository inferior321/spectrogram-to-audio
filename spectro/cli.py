"""Headless conversion, for scripting or for machines with no display.

    ./run.sh --cli image.png -o out.mp3 --source "Color (classic)" --scale Mel
    ./run.sh --cli folder/*.png --outdir audio/ --source viridis
    ./run.sh --cli --list-sources
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import audio_io, colormap, core, sources
from .settings import (
    ORIENTATIONS, FREQ_SCALES, LEVEL_MAPPINGS, OUTPUT_FORMATS, PHASE_INITS, QUALITY_PRESETS,
    WINDOW_TYPES, Settings,
)


def build_parser() -> argparse.ArgumentParser:
    st = Settings()
    p = argparse.ArgumentParser(
        prog="spectrogram-to-audio --cli",
        description="Turn spectrogram images into audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("images", nargs="*", type=Path, help="input image(s)")
    p.add_argument("-o", "--output", type=Path, help="output file (single input only)")
    p.add_argument("--outdir", type=Path, help="output folder for multiple inputs")

    g = p.add_argument_group("frequency axis")
    g.add_argument("--scale", choices=FREQ_SCALES, default=st.freq_scale)
    g.add_argument("--fmin", type=float, default=st.min_freq)
    g.add_argument("--fmax", type=float, default=st.max_freq)

    g = p.add_argument_group("colour scheme")
    g.add_argument("--source", default=None,
                   help="the tool and scheme that drew the image, by catalogue "
                        "label or any unambiguous part of one, e.g. "
                        "'classic' or 'viridis'; see --list-sources")
    g.add_argument("--scheme", default=st.color_scheme,
                   help="the gradient directly, if you would rather name it; "
                        "see --list-schemes")
    g.add_argument("--list-sources", action="store_true",
                   help="print the source catalogue and exit")
    g.add_argument("--list-schemes", action="store_true",
                   help="print every known colour gradient and exit")
    g.add_argument("--orientation", choices=ORIENTATIONS, default=ORIENTATIONS[0],
                   help="which way the picture is laid out")
    g.add_argument("--flip", action="store_true",
                   help="swap loud and quiet (the scheme normally decides this)")

    g = p.add_argument_group("levels")
    g.add_argument("--mapping", choices=LEVEL_MAPPINGS, default=st.level_mapping)
    g.add_argument("--gain", type=float, default=st.gain_db)
    g.add_argument("--range", dest="range_db", type=float, default=st.range_db)
    g.add_argument("--high-boost", type=float, default=st.high_boost_db_per_dec)
    g.add_argument("--gate", type=float, default=st.floor_gate)
    g.add_argument("--denoise", type=float, default=st.denoise_db,
                   help="dB of background reduction; 0 is off")
    g.add_argument("--denoise-pct", type=float, default=st.denoise_percentile,
                   help="percentile over time treated as each bin's noise floor")

    g = p.add_argument_group("fft")
    g.add_argument("--window", type=int, default=st.window_size)
    g.add_argument("--window-type", choices=WINDOW_TYPES, default=st.window_type)
    g.add_argument("--padding", type=int, default=st.zero_padding)
    g.add_argument("--overlap", type=int, default=st.overlap)

    g = p.add_argument_group("region")
    g.add_argument("--from", dest="region_from", type=float, default=0.0,
                   help="start of the time region to convert, as a fraction 0..1")
    g.add_argument("--to", dest="region_to", type=float, default=1.0,
                   help="end of the time region to convert, as a fraction 0..1")

    g = p.add_argument_group("time")
    g.add_argument("--rate", type=int, default=st.sample_rate)
    g.add_argument("--duration", default="", help="m:ss or seconds; blank = infer")
    g.add_argument("--stretch", type=float, default=st.time_stretch)

    g = p.add_argument_group("phase")
    g.add_argument("--quality", choices=list(QUALITY_PRESETS), default=st.quality,
                   help="sets iterations, momentum and starting phase together; "
                        "any of the three given explicitly wins")
    g.add_argument("--phase-init", choices=PHASE_INITS, default=None)
    g.add_argument("--iters", type=int, default=None)
    g.add_argument("--momentum", type=float, default=None)
    g.add_argument("--seed", type=int, default=st.random_seed)

    g = p.add_argument_group("output")
    g.add_argument("--format", choices=OUTPUT_FORMATS, default=st.output_format)
    g.add_argument("--bitrate", type=int, default=st.mp3_bitrate_kbps)
    g.add_argument("--no-normalize", action="store_true")
    g.add_argument("-q", "--quiet", action="store_true")
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    st = Settings()

    from .gui import parse_duration          # one shared duration parser

    st.freq_scale, st.min_freq, st.max_freq = args.scale, args.fmin, args.fmax
    st.level_mapping, st.gain_db, st.range_db = args.mapping, args.gain, args.range_db
    st.high_boost_db_per_dec, st.floor_gate = args.high_boost, args.gate
    st.color_scheme = args.scheme
    if args.source:
        matches = [x for x in sources.catalogue()
                   if not x.is_separator and args.source.lower() in x.label.lower()]
        if len(matches) == 1:
            st.color_scheme = matches[0].scheme
            for key, value in matches[0].overrides.items():
                setattr(st, key, value)
        elif not matches:
            print(f"no source matching {args.source!r}; try --list-sources",
                  file=sys.stderr)
        else:
            print(f"{args.source!r} matches {len(matches)} sources: "
                  + ", ".join(m.label for m in matches[:6]), file=sys.stderr)
    st.orientation = args.orientation
    st.flip_polarity = args.flip
    st.denoise_db, st.denoise_percentile = args.denoise, args.denoise_pct
    st.preview_start, st.preview_end = args.region_from, args.region_to
    st.window_size, st.window_type = args.window, args.window_type
    st.zero_padding, st.overlap = args.padding, args.overlap
    st.sample_rate, st.time_stretch = args.rate, args.stretch
    st.duration_s = parse_duration(args.duration)
    # Quality is shorthand for three settings; anything named explicitly wins.
    st.quality = args.quality
    for key, value in QUALITY_PRESETS.get(args.quality, {}).items():
        setattr(st, key, value)
    if args.iters is not None:
        st.gl_iterations = args.iters
    if args.momentum is not None:
        st.gl_momentum = args.momentum
    if args.phase_init is not None:
        st.phase_init = args.phase_init
    st.random_seed = args.seed
    st.output_format, st.mp3_bitrate_kbps = args.format, args.bitrate
    st.normalize = not args.no_normalize
    return st


def main() -> int:
    args = build_parser().parse_args()
    if args.list_sources:
        for src in sources.catalogue():
            if src.is_separator:
                print(f"\n{src.label}")
            else:
                tag = "  [exact]" if src.accuracy == "exact" else ""
                print(f"  {src.label}{tag}")
        return 0
    if args.list_schemes:
        for name in colormap.available():
            print(name)
        return 0
    st = settings_from_args(args)

    if args.output and len(args.images) > 1:
        print("--output takes a single input; use --outdir instead", file=sys.stderr)
        return 2

    failures = 0
    for path in args.images:
        if not path.is_file():
            print(f"{path}: not found", file=sys.stderr)
            failures += 1
            continue
        this = st.copy()
        rgb = core.apply_orientation(core.load_image(str(path)), this)

        def report(frac: float, msg: str) -> None:
            if not args.quiet:
                print(f"\r  {frac * 100:5.1f}%  {msg:<44}", end="", flush=True)

        res = core.convert(core.to_level(rgb, this), this, report)
        if not args.quiet:
            print()

        if args.output:
            dest = args.output
        else:
            out_dir = args.outdir or path.parent
            dest = out_dir / f"{path.stem}.{this.output_format}"
        written = audio_io.export(res.audio, res.sample_rate, dest,
                                  this.output_format, this.mp3_bitrate_kbps,
                                  this.bit_depth)
        if not args.quiet:
            print(f"  {path.name} -> {written}  "
                  f"({res.duration_s:.2f} s, peak {res.peak_dbfs:.1f} dBFS)")
    return 1 if failures else 0
