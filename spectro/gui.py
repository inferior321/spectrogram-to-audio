"""The main window.

Layout: image + waveform on the left, every setting on the right, a log and a
batch tab underneath.  Conversion runs on a worker thread so the window never
freezes and can be cancelled mid-run.
"""

from __future__ import annotations

import dataclasses
import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from . import audio_io, colormap, core, sources
from .settings import (
    BIT_DEPTHS, FREQ_SCALES, LEVEL_MAPPINGS, ORIENTATIONS, OUTPUT_FORMATS,
    PHASE_INITS, QUALITY_PRESETS, WINDOW_TYPES, Settings,
)
from .widgets import ImageView, Section, SliderSpin, WaveformView

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)"


def parse_duration(text: str) -> float:
    """Accept '1:23.5', '83.5', '1m23s' or blank.  Blank/0 means 'infer'."""
    text = text.strip().lower().replace("m", ":").replace("s", "")
    if not text:
        return 0.0
    try:
        if ":" in text:
            parts = [p for p in text.split(":") if p != ""]
            total = 0.0
            for part in parts:
                total = total * 60.0 + float(part)
            return total
        return float(text)
    except ValueError:
        return 0.0


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return ""
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}" if m else f"{s:.2f}"


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

class ConvertWorker(QObject):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, level: np.ndarray, st: Settings) -> None:
        super().__init__()
        self._level, self._st = level, st
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        def report(frac: float, msg: str) -> None:
            if self._cancel:
                raise core.Cancelled()
            self.progress.emit(frac, msg)

        try:
            self.finished.emit(core.convert(self._level, self._st, report))
        except core.Cancelled:
            self.failed.emit("Cancelled.")
        except Exception:
            self.failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Spectrogram to Audio")
        self.resize(1360, 900)
        self.setAcceptDrops(True)

        self.image_path: Path | None = None
        self.rgb: np.ndarray | None = None
        self.result: core.Result | None = None
        self.player = audio_io.Player()
        self._thread: QThread | None = None
        self._worker: ConvertWorker | None = None

        self._build_ui()
        self._build_menu()
        self.apply_settings(Settings())
        self.log("Ready. Open a spectrogram image to begin.")
        if not self.player.available:
            self.log(f"Playback unavailable ({self.player.error}). "
                     "Export still works.", "warn")

        self._playtimer = QTimer(self)
        self._playtimer.setInterval(50)
        self._playtimer.timeout.connect(self._tick_playhead)

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 540])
        self.setCentralWidget(splitter)

        self.status = self.statusBar()
        self.status.showMessage("Ready")

    def _build_left(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 4, 8)

        bar = QHBoxLayout()
        self.btn_open = QPushButton("Open image…")
        self.btn_open.clicked.connect(self.open_image)
        self.lbl_file = QLabel("no file")
        # no colour override: palette(mid) is a border colour and on a
        # dark theme it lands 7 luminance units from the background,
        # which is invisible. The default text colour is the only one
        # guaranteed to contrast with the background it sits on.
        self.lbl_file.setStyleSheet("")
        # Long names are elided rather than allowed to push the window wider.
        self.lbl_file.setSizePolicy(QSizePolicy.Policy.Ignored,
                                    QSizePolicy.Policy.Preferred)
        self.lbl_file.setMinimumWidth(60)
        bar.addWidget(self.btn_open)
        bar.addWidget(self.lbl_file, 1)
        lay.addLayout(bar)

        self.image_view = ImageView()
        self.image_view.hovered.connect(self._on_hover)
        self.image_view.regionChanged.connect(self._on_region)
        lay.addWidget(self.image_view, 1)

        self.lbl_hover = QLabel("Hover the image to read off frequency and time.")
        self.lbl_hover.setStyleSheet("font-size: 11px;")
        lay.addWidget(self.lbl_hover)

        self.waveform = WaveformView()
        lay.addWidget(self.waveform)

        play = QHBoxLayout()
        self.btn_convert = QPushButton("Convert")
        self.btn_convert.setDefault(True)
        self.btn_convert.clicked.connect(self.start_convert)
        self.btn_convert.setEnabled(False)
        self.btn_preview = QPushButton("Preview region")
        self.btn_preview.setToolTip(
            "Convert only the selected part of the image at draft quality. "
            "Drag across the image to choose a few seconds; with nothing "
            "selected this previews the first few seconds.")
        self.btn_preview.clicked.connect(self.start_preview)
        self.btn_preview.setEnabled(False)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_convert)
        self.btn_cancel.setEnabled(False)
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        self.btn_export = QPushButton("Export…")
        self.btn_export.clicked.connect(self.export_audio)
        self.btn_export.setEnabled(False)
        for b in (self.btn_convert, self.btn_preview, self.btn_cancel,
                  self.btn_play, self.btn_export):
            play.addWidget(b)
        play.addStretch(1)
        lay.addLayout(play)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%  —  idle")
        lay.addWidget(self.progress)

        self.tabs_bottom = QTabWidget()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(150)
        self.tabs_bottom.addTab(self.log_view, "Log")
        self.tabs_bottom.addTab(self._build_batch(), "Batch")
        lay.addWidget(self.tabs_bottom)
        return page

    def _build_batch(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        note = QLabel(
            "Convert every image in a folder using exactly the settings on the "
            "right. Every image is read the same way, so put images from one "
            "source and one frequency axis in a folder together.")
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 11px;")
        lay.addWidget(note)

        row = QHBoxLayout()
        self.batch_in = QLineEdit()
        self.batch_in.setPlaceholderText("input folder of images…")
        b1 = QPushButton("Browse…")
        b1.clicked.connect(lambda: self._pick_dir(self.batch_in))
        row.addWidget(QLabel("Input:"))
        row.addWidget(self.batch_in, 1)
        row.addWidget(b1)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        self.batch_out = QLineEdit()
        self.batch_out.setPlaceholderText("output folder…")
        b2 = QPushButton("Browse…")
        b2.clicked.connect(lambda: self._pick_dir(self.batch_out))
        row2.addWidget(QLabel("Output:"))
        row2.addWidget(self.batch_out, 1)
        row2.addWidget(b2)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_batch = QPushButton("Run batch")
        self.btn_batch.clicked.connect(self.run_batch)
        row3.addStretch(1)
        row3.addWidget(self.btn_batch)
        lay.addLayout(row3)
        lay.addStretch(1)
        return page

    def _pick_dir(self, line: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose folder", line.text() or str(Path.home()))
        if d:
            line.setText(d)

    def _build_right(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(430)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        top = QHBoxLayout()
        btn_reset = QPushButton("Reset to defaults")
        btn_reset.setToolTip("Restore the Audacity dialog values this project "
                             "was built around: Grayscale, Linear, 1–20000 Hz, "
                             "Gain 20, Range 80, Hann 2048, zero padding 2.")
        btn_reset.clicked.connect(self.reset_defaults)
        top.addStretch(1)
        top.addWidget(btn_reset)
        lay.addLayout(top)

        # ---- source image -------------------------------------------------
        s = Section("Source image")
        self.cmb_source = QComboBox()
        self.cmb_source.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_source.setMinimumContentsLength(14)
        self._fill_sources()
        self.cmb_source.currentIndexChanged.connect(self._on_source_changed)
        s.add_row("Came from:", self.cmb_source,
                  "Which tool drew the image and which colour scheme it was "
                  "set to. Nothing is guessed - pick the one you exported. "
                  "The frequency axis is chosen separately, below.")
        self.cmb_orientation = QComboBox()
        self.cmb_orientation.addItems(ORIENTATIONS)
        self.cmb_orientation.currentTextChanged.connect(self._on_orientation_changed)
        s.add_row("Layout:", self.cmb_orientation,
                  "Most tools draw time along the bottom and frequency up the "
                  "side. Some can draw it rotated — ffmpeg's showspectrum will, "
                  "and Spectrogrammer exposes it as Orientation. A rotated "
                  "image read as a normal one is not slightly wrong, it is "
                  "nonsense, because frequency becomes time. Both directions of "
                  "rotation are offered because which one a tool uses is not "
                  "worth guessing: pick the one whose preview looks like a "
                  "spectrogram.")
        self.lbl_source = QLabel()
        self.lbl_source.setWordWrap(True)
        self.lbl_source.setStyleSheet("font-size: 11px;")
        s.add_wide(self.lbl_source)
        self.chk_flip = QCheckBox("Flip polarity (swap loud and quiet)")
        s.add_wide(self.chk_flip)
        s.add_note("The scheme already knows which end is loud. Use this only "
                   "when an image is the wrong way round anyway.")
        self.cmb_channel = QComboBox()
        self.cmb_channel.addItems(["Luminance", "Max channel", "Red", "Green", "Blue"])
        s.add_row("Read pixels as:", self.cmb_channel,
                  "Greyscale schemes only: how a pixel becomes one number.")
        self.sld_crop_l = SliderSpin(0, 400, 0, 1, 0, " px")
        self.sld_crop_r = SliderSpin(0, 400, 0, 1, 0, " px")
        self.sld_crop_t = SliderSpin(0, 400, 0, 1, 0, " px")
        self.sld_crop_b = SliderSpin(0, 400, 0, 1, 0, " px")
        s.add_row("Crop left:", self.sld_crop_l, "Trim axis labels, rulers or borders.")
        s.add_row("Crop right:", self.sld_crop_r)
        s.add_row("Crop top:", self.sld_crop_t)
        s.add_row("Crop bottom:", self.sld_crop_b)
        lay.addWidget(s)
        self.sec_source = s

        # ---- frequency axis ----------------------------------------------
        s = Section("Frequency axis")
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(FREQ_SCALES)
        s.add_row("Scale:", self.cmb_scale,
                  "Must match the scale the image was drawn with. Getting this "
                  "wrong warps pitch in a way no speed change can fix.")
        self.sld_fmin = SliderSpin(0, 2000, 1, 1, 0, " Hz")
        self.sld_fmax = SliderSpin(1000, 48000, 20000, 100, 0, " Hz")
        s.add_row("Min frequency:", self.sld_fmin, "Frequency of the BOTTOM pixel row.")
        s.add_row("Max frequency:", self.sld_fmax, "Frequency of the TOP pixel row.")
        s.add_note("Set these to the values the image was exported with, not to "
                   "where the visible content stops.")
        lay.addWidget(s)
        self.sec_freq = s

        # ---- levels -------------------------------------------------------
        s = Section("Levels / colour mapping")
        self.cmb_mapping = QComboBox()
        self.cmb_mapping.addItems(LEVEL_MAPPINGS)
        s.add_row("Mapping:", self.cmb_mapping,
                  "How brightness becomes loudness. The Audacity option reproduces "
                  "that program's exact gain/range formula.")
        self.sld_gain = SliderSpin(-60, 60, 20, 1, 0, " dB")
        self.sld_range = SliderSpin(10, 160, 80, 1, 0, " dB")
        self.sld_boost = SliderSpin(0, 60, 0, 1, 0, " dB/dec")
        self.sld_gate = SliderSpin(0.0, 0.9, 0.0, 0.01, 2, "")
        s.add_row("Gain:", self.sld_gain)
        s.add_row("Range:", self.sld_range)
        s.add_row("High boost:", self.sld_boost,
                  "Undo Audacity's high-frequency tilt. Leave at 0 unless the "
                  "export used it.")
        s.add_row("Noise gate:", self.sld_gate,
                  "Force levels below this to true silence. Useful when the "
                  "background is a light grey rather than pure white.")
        self.lbl_calib = QLabel()
        self.lbl_calib.setWordWrap(True)
        self.lbl_calib.setStyleSheet("font-size: 11px;")
        s.add_wide(self.lbl_calib)
        lay.addWidget(s)
        self.sec_level = s

        # ---- FFT ----------------------------------------------------------
        s = Section("Analysis (FFT)")
        self.cmb_window_size = QComboBox()
        self.cmb_window_size.addItems([str(2 ** i) for i in range(6, 16)])
        s.add_row("Window size:", self.cmb_window_size,
                  "Match the export. This sets frequency resolution.")
        self.cmb_window_type = QComboBox()
        self.cmb_window_type.addItems(WINDOW_TYPES)
        s.add_row("Window type:", self.cmb_window_type,
                  "Set this to whatever drew the image — Hann for Audacity. "
                  "A Gaussian window has no sidelobes (-107 dB against Hann's "
                  "-31 dB) so it sounds smoother and less robotic, but measured "
                  "against known audio it is also less accurate (4.1 dB of "
                  "error against Hann's 0.8), because it answers a mismatched "
                  "question by smoothing, and that takes real detail with the "
                  "artefacts. A fair trade only if you prefer the sound.")
        self.cmb_padding = QComboBox()
        self.cmb_padding.addItems(["1", "2", "4", "8"])
        s.add_row("Zero padding:", self.cmb_padding,
                  "FFT size = window size × this, exactly as in Audacity.")
        self.lbl_bins = QLabel()
        self.lbl_bins.setWordWrap(True)
        self.lbl_bins.setStyleSheet("font-size: 11px;")
        s.add_wide(self.lbl_bins)
        self.cmb_overlap = QComboBox()
        self.cmb_overlap.addItems(["2", "4", "8", "16"])
        s.add_row("Overlap factor:", self.cmb_overlap,
                  "Hop = window ÷ this. Higher is smoother and slower. This one "
                  "is a resynthesis choice and has no equivalent in the export "
                  "dialog — 4 is a good default.")
        lay.addWidget(s)
        self.sec_fft = s

        # ---- time ---------------------------------------------------------
        s = Section("Time / duration")
        self.cmb_rate = QComboBox()
        self.cmb_rate.setEditable(True)
        self.cmb_rate.addItems(["8000", "16000", "22050", "32000", "44100", "48000", "96000"])
        s.add_row("Sample rate:", self.cmb_rate, "Hz")
        self.edit_duration = QLineEdit()
        self.edit_duration.setPlaceholderText("blank = infer from image width  (m:ss or seconds)")
        s.add_row("Duration:", self.edit_duration,
                  "Type 1:23.5, or 83.5, or leave blank to let the program guess.")
        self.sld_stretch = SliderSpin(0.10, 4.00, 1.00, 0.01, 2, "×")
        s.add_row("Time stretch:", self.sld_stretch,
                  "By-ear speed control. Multiplies the duration above; also "
                  "shifts pitch, exactly like changing tape speed.")
        self.lbl_time = QLabel()
        self.lbl_time.setWordWrap(True)
        self.lbl_time.setStyleSheet("font-size: 11px;")
        s.add_wide(self.lbl_time)
        lay.addWidget(s)
        self.sec_time = s

        # ---- phase --------------------------------------------------------
        s = Section("Phase reconstruction")
        self.cmb_quality = QComboBox()
        self.cmb_quality.addItems(list(QUALITY_PRESETS))
        self.cmb_quality.activated.connect(self._apply_quality)
        s.add_row("Quality:", self.cmb_quality,
                  "Sets iterations, momentum and starting phase together. "
                  "Choosing anything below moves this to Custom.")
        self.sld_iters = SliderSpin(0, 1000, 100, 1, 0, " iters")
        self.sld_momentum = SliderSpin(0.0, 1.0, 0.99, 0.01, 2, "")
        self.cmb_phase_init = QComboBox()
        self.cmb_phase_init.addItems(PHASE_INITS)
        self.sld_seed = SliderSpin(0, 9999, 0, 1, 0, "")
        s.add_row("Iterations:", self.sld_iters,
                  "More is cleaner and slower. Returns diminish past ~300.")
        s.add_row("Momentum:", self.sld_momentum,
                  "0 = classic Griffin-Lim. 0.99 = Fast Griffin-Lim, about three "
                  "times quicker for the same quality. Values above 1 are unstable "
                  "and are clamped.")
        s.add_row("Initial phase:", self.cmb_phase_init,
                  "PGHI estimates a starting phase from the spectrogram's own "
                  "gradients instead of from noise. On a true spectrogram it "
                  "beats random phase by around 47x. On an image it is far "
                  "less certain, because those gradients have been blurred by "
                  "resampling and flattened by 8-bit colour — measured on this "
                  "project's references it still helps with a Hann window, but "
                  "is close to a wash — 0.76 dB against 0.75 dB measured "
                  "against known audio. It costs a couple of seconds and does "
                  "no harm, so it stays on by default.")
        s.add_row("Random seed:", self.sld_seed, "Same seed gives the same output.")
        s.add_note("A picture stores no phase, so it has to be invented. Even a "
                   "perfect run keeps a slight metallic or watery quality — that is "
                   "the method's limit, not a setting you have got wrong.")
        lay.addWidget(s)
        self.sec_phase = s

        # ---- denoise ------------------------------------------------------
        s = Section("Denoise")
        self.sld_denoise = SliderSpin(0, 40, 0, 1, 0, " dB")
        s.add_row("Reduce background:", self.sld_denoise,
                  "0 is off. Attenuates each frequency bin by how close it sits "
                  "to its own noise floor.")
        self.sld_denoise_pct = SliderSpin(1, 60, 20, 1, 0, "th pct")
        s.add_row("Floor estimate:", self.sld_denoise_pct,
                  "Which percentile over time counts as 'the background' for "
                  "each bin. Lower assumes the background is quieter.")
        s.add_note("Applied to the magnitudes before phase reconstruction, so "
                   "Griffin-Lim never has to invent phase for hiss - which is "
                   "where much of the watery quality comes from.")
        lay.addWidget(s)
        self.sec_denoise = s

        # ---- output -------------------------------------------------------
        s = Section("Output")
        self.chk_normalize = QCheckBox("Normalize peak")
        s.add_wide(self.chk_normalize)
        self.sld_target = SliderSpin(-30.0, 0.0, -1.0, 0.5, 1, " dBFS")
        s.add_row("Peak target:", self.sld_target)
        self.sld_fade = SliderSpin(0, 500, 10, 1, 0, " ms")
        s.add_row("Fade in/out:", self.sld_fade, "Removes the click at each end.")
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(OUTPUT_FORMATS)
        s.add_row("Format:", self.cmb_format)
        self.cmb_bitrate = QComboBox()
        self.cmb_bitrate.addItems(["128", "192", "256", "320"])
        s.add_row("MP3 bitrate:", self.cmb_bitrate, "kbps")
        self.cmb_depth = QComboBox()
        self.cmb_depth.addItems(BIT_DEPTHS)
        s.add_row("WAV/FLAC depth:", self.cmb_depth)
        lay.addWidget(s)
        self.sec_out = s

        lay.addStretch(1)
        scroll.setWidget(inner)

        for w in (self.sld_iters, self.sld_momentum):
            w.valueChanged.connect(self._to_custom)
        self.cmb_phase_init.currentTextChanged.connect(self._to_custom)

        # Live readouts.
        for w in (self.sld_gain, self.sld_range):
            w.valueChanged.connect(self._update_readouts)
        for w in (self.sld_stretch,):
            w.valueChanged.connect(self._update_readouts)
        self.cmb_mapping.currentTextChanged.connect(self._update_readouts)
        self.cmb_rate.currentTextChanged.connect(self._update_readouts)
        self.edit_duration.textChanged.connect(self._update_readouts)
        self.cmb_window_size.currentTextChanged.connect(self._update_readouts)
        self.cmb_overlap.currentTextChanged.connect(self._update_readouts)
        self.cmb_format.currentTextChanged.connect(self._update_readouts)
        self.sld_denoise.valueChanged.connect(self._update_readouts)
        self.cmb_padding.currentTextChanged.connect(self._update_readouts)
        return scroll

    # -- source catalogue --------------------------------------------------

    def _fill_sources(self) -> None:
        """Rebuild the dropdown, keeping headings unselectable."""
        current = getattr(self, "_current_scheme", Settings().color_scheme)
        self.cmb_source.blockSignals(True)
        self.cmb_source.clear()
        model = self.cmb_source.model()
        for i, src in enumerate(sources.catalogue()):
            self.cmb_source.addItem(src.label)
            if src.is_separator:
                # A heading is a row, not a choice.
                item = model.item(i)
                item.setEnabled(False)
                item.setData(0, Qt.ItemDataRole.UserRole - 1)
        self.cmb_source.blockSignals(False)
        self._select_scheme(current)

    def _select_scheme(self, scheme: str) -> None:
        """Point the dropdown at whichever entry uses this gradient."""
        self._current_scheme = scheme
        label = sources.label_for_scheme(scheme)
        self.cmb_source.blockSignals(True)
        if label is not None:
            self.cmb_source.setCurrentText(label)
        self.cmb_source.blockSignals(False)
        self._describe_source()

    def _on_source_changed(self, index: int) -> None:
        src = sources.find(self.cmb_source.itemText(index))
        if src is None:
            return
        self._current_scheme = src.scheme
        # Some sources also pin the level mapping, because the tool documents
        # its own calibration - SoX states 120 dB of range topping out at
        # 0 dBFS, so there is nothing to guess about how bright means how loud.
        widgets = {
            "level_mapping": self.cmb_mapping,
            "min_freq": self.sld_fmin,
            "max_freq": self.sld_fmax,
            "range_db": self.sld_range,
            "gain_db": self.sld_gain,
        }

        def put(key: str, value: object) -> None:
            widget = widgets.get(key)
            if widget is None:
                return
            if isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            else:
                widget.set_value(float(value))

        # Undo what the LAST source imposed before applying this one, so that
        # picking SoX and then Spek does not leave Spek wearing SoX's 120 dB
        # range.  Only those keys are touched: anything the user set by hand is
        # theirs and survives a change of source.
        defaults = Settings()
        for key in getattr(self, "_source_keys", set()) - set(src.overrides):
            put(key, getattr(defaults, key))
        for key, value in src.overrides.items():
            put(key, value)
        self._source_keys = set(src.overrides)
        self._describe_source()
        self._update_readouts()

    def _describe_source(self) -> None:
        # The dropdown is filled while the panel is still being built, so the
        # widgets it talks to may not exist yet.
        if not hasattr(self, "lbl_source"):
            return
        src = sources.find(self.cmb_source.currentText())
        if hasattr(self, "cmb_channel"):
            self.cmb_channel.setEnabled(self._current_scheme in colormap.BUILTIN)
        if src is None:
            self.lbl_source.setText("")
            return
        badge = "Exact table." if src.accuracy == "exact" else ""
        self.lbl_source.setText(" ".join(x for x in (badge, src.note) if x))

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        act_open = QAction("&Open image…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_image)
        act_export = QAction("&Export audio…", self)
        act_export.setShortcut(QKeySequence.StandardKey.Save)
        act_export.triggered.connect(self.export_audio)
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        m.addAction(act_open)
        m.addAction(act_export)
        m.addSeparator()
        m.addAction(act_quit)

        m2 = self.menuBar().addMenu("&Convert")
        act_run = QAction("&Convert now", self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(self.start_convert)
        act_reset = QAction("&Reset to defaults", self)
        act_reset.triggered.connect(self.reset_defaults)
        for a in (act_run, act_reset):
            m2.addAction(a)

        m3 = self.menuBar().addMenu("&Help")
        act_about = QAction("About / how this works", self)
        act_about.triggered.connect(self.show_about)
        m3.addAction(act_about)

    # -- settings <-> widgets ---------------------------------------------

    def current_settings(self) -> Settings:
        st = Settings()
        st.orientation = self.cmb_orientation.currentText()
        st.color_scheme = self._current_scheme
        st.flip_polarity = self.chk_flip.isChecked()
        st.channel_mode = self.cmb_channel.currentText()
        st.crop_left = int(self.sld_crop_l.value())
        st.crop_right = int(self.sld_crop_r.value())
        st.crop_top = int(self.sld_crop_t.value())
        st.crop_bottom = int(self.sld_crop_b.value())

        st.freq_scale = self.cmb_scale.currentText()
        st.min_freq = float(self.sld_fmin.value())
        st.max_freq = float(self.sld_fmax.value())

        st.level_mapping = self.cmb_mapping.currentText()
        st.gain_db = float(self.sld_gain.value())
        st.range_db = float(self.sld_range.value())
        st.high_boost_db_per_dec = float(self.sld_boost.value())
        st.floor_gate = float(self.sld_gate.value())

        st.window_size = int(self.cmb_window_size.currentText())
        st.window_type = self.cmb_window_type.currentText()
        st.zero_padding = int(self.cmb_padding.currentText())
        st.overlap = int(self.cmb_overlap.currentText())

        try:
            st.sample_rate = max(1000, int(float(self.cmb_rate.currentText())))
        except ValueError:
            st.sample_rate = 44100
        st.duration_s = parse_duration(self.edit_duration.text())
        st.time_stretch = float(self.sld_stretch.value())

        st.quality = self.cmb_quality.currentText()
        st.gl_iterations = int(self.sld_iters.value())
        st.gl_momentum = float(self.sld_momentum.value())
        st.phase_init = self.cmb_phase_init.currentText()
        st.random_seed = int(self.sld_seed.value())

        st.denoise_db = float(self.sld_denoise.value())
        st.denoise_percentile = float(self.sld_denoise_pct.value())
        sel = self.image_view.selection()
        st.preview_start, st.preview_end = sel if sel else (0.0, 1.0)
        st.normalize = self.chk_normalize.isChecked()
        st.target_dbfs = float(self.sld_target.value())
        st.fade_ms = float(self.sld_fade.value())
        st.output_format = self.cmb_format.currentText()
        st.mp3_bitrate_kbps = int(self.cmb_bitrate.currentText())
        st.bit_depth = self.cmb_depth.currentText()
        return st

    def apply_settings(self, st: Settings) -> None:
        self.cmb_orientation.setCurrentText(st.orientation)
        self._select_scheme(st.color_scheme)
        self.chk_flip.setChecked(st.flip_polarity)
        self.cmb_channel.setCurrentText(st.channel_mode)
        self.sld_crop_l.set_value(st.crop_left)
        self.sld_crop_r.set_value(st.crop_right)
        self.sld_crop_t.set_value(st.crop_top)
        self.sld_crop_b.set_value(st.crop_bottom)

        self.cmb_scale.setCurrentText(st.freq_scale)
        self.sld_fmin.set_value(st.min_freq)
        self.sld_fmax.set_value(st.max_freq)

        self.cmb_mapping.setCurrentText(st.level_mapping)
        self.sld_gain.set_value(st.gain_db)
        self.sld_range.set_value(st.range_db)
        self.sld_boost.set_value(st.high_boost_db_per_dec)
        self.sld_gate.set_value(st.floor_gate)

        self.cmb_window_size.setCurrentText(str(st.window_size))
        self.cmb_window_type.setCurrentText(st.window_type)
        self.cmb_padding.setCurrentText(str(st.zero_padding))
        self.cmb_overlap.setCurrentText(str(st.overlap))

        self.cmb_rate.setCurrentText(str(st.sample_rate))
        self.edit_duration.setText(format_duration(st.duration_s))
        self.sld_stretch.set_value(st.time_stretch)

        self.cmb_quality.setCurrentText(st.quality)
        self.sld_iters.set_value(st.gl_iterations)
        self.sld_momentum.set_value(st.gl_momentum)
        self.cmb_phase_init.setCurrentText(st.phase_init)
        self.sld_seed.set_value(st.random_seed)

        self.sld_denoise.set_value(st.denoise_db)
        self.sld_denoise_pct.set_value(st.denoise_percentile)
        self.chk_normalize.setChecked(st.normalize)
        self.sld_target.set_value(st.target_dbfs)
        self.sld_fade.set_value(st.fade_ms)
        self.cmb_format.setCurrentText(st.output_format)
        self.cmb_bitrate.setCurrentText(str(st.mp3_bitrate_kbps))
        self.cmb_depth.setCurrentText(st.bit_depth)
        self._update_readouts()

    def _apply_quality(self, *_: object) -> None:
        """Quality is a shorthand for three phase settings at once."""
        values = QUALITY_PRESETS.get(self.cmb_quality.currentText(), {})
        for key, value in values.items():
            if key == "gl_iterations":
                self.sld_iters.set_value(value)
            elif key == "gl_momentum":
                self.sld_momentum.set_value(value)
            elif key == "phase_init":
                self.cmb_phase_init.setCurrentText(value)
        self._update_readouts()

    def _to_custom(self, *_: object) -> None:
        """Any hand edit of a phase control means Quality no longer describes it."""
        current = self.cmb_quality.currentText()
        if current == "Custom":
            return
        wanted = QUALITY_PRESETS.get(current, {})
        actual = {
            "gl_iterations": int(self.sld_iters.value()),
            "gl_momentum": round(float(self.sld_momentum.value()), 2),
            "phase_init": self.cmb_phase_init.currentText(),
        }
        if any(actual.get(k) != v for k, v in wanted.items()):
            self.cmb_quality.setCurrentText("Custom")

    @staticmethod
    def _suggest_window(rows: int, st: Settings) -> str:
        """Which window size and padding would match this many image rows.

        The transform wants about 2 bins per row, so n_fft = 2 * rows.  The
        window carries that, and the padding only multiplies it, so the padding
        is left alone and the window is rounded to the nearest power of two
        that gets there.
        """
        target = max(64, 2 * rows)
        window = max(64, min(32768, target // max(1, st.zero_padding)))
        window = 1 << max(6, min(15, round(window).bit_length() - 1))
        if st.zero_padding != 1 and window * st.zero_padding != target:
            return f"window {2 * rows} with padding 1"
        return f"window {window}"

    def _update_readouts(self, *_: object) -> None:
        st = self.current_settings()

        # Name the ends by the colours the CHOSEN scheme actually uses. This
        # said "white = quiet, black = loud" regardless, which is right for
        # Audacity's Grayscale and backwards for ffmpeg, SoX and Spek.
        quiet, loud = colormap.scheme_ends(st.color_scheme)
        if st.flip_polarity:
            quiet, loud = loud, quiet
        if st.level_mapping == "Audacity dB (gain/range)":
            self.lbl_calib.setText(
                f"Quiet end ({quiet}) = {-st.range_db - st.gain_db:.0f} dBFS, "
                f"loud end ({loud}) = {-st.gain_db:.0f} dBFS. "
                "Raising Gain makes the result quieter overall; raising Range "
                "lifts more of the quiet detail.")
        elif st.level_mapping == "Plain dB range":
            self.lbl_calib.setText(
                f"Quiet end ({quiet}) = {-st.range_db:.0f} dBFS, "
                f"loud end ({loud}) = 0 dBFS. Gain is ignored.")
        else:
            self.lbl_calib.setText(
                f"Brightness is used directly, {quiet} through {loud}; "
                "Gain and Range are ignored.")

        full = self.rgb.shape[1] - st.crop_left - st.crop_right if self.rgb is not None else 0
        if full > 0:
            frac = core.preview_fraction(st)
            width = max(4, int(round(full * frac)))
            frames = core.infer_frame_count(width, st)
            dur = core.frames_to_duration(frames, st)
            src = "from your duration" if st.duration_s > 0 else "inferred (1 column = 1 frame)"
            region = "" if frac >= 1.0 else f"  [region: {frac * 100:.0f}% of the image]"
            self.lbl_time.setText(
                f"{width} columns → {frames} frames → {dur:.2f} s at "
                f"{st.sample_rate} Hz, hop {st.hop} ({src}).{region}")
        else:
            self.lbl_time.setText("Load an image to see the resulting length.")

        bins = st.n_fft // 2 + 1
        rows = (self.rgb.shape[0] - st.crop_top - st.crop_bottom) if self.rgb is not None else 0
        if rows:
            ratio = bins / max(1, rows)
            if 0.7 <= ratio <= 1.6:
                note = "well matched"
            else:
                # Name the setting that actually fixes it.  This used to say
                # "try a smaller zero padding" whatever was wrong, which is
                # useless advice when the padding is already 1 and the window
                # is the thing that is out - the common mistake, because the
                # window size is set by the image's HEIGHT and it is natural to
                # reach for its width.
                want = self._suggest_window(rows, st)
                trouble = ("mostly interpolation" if ratio > 1.6
                           else "coarser than the image, detail discarded")
                note = f"{trouble} - try {want}"
            self.lbl_bins.setText(f"{rows} image rows -> {bins} FFT bins "
                                  f"({ratio:.1f}x, {note}).")
        else:
            self.lbl_bins.setText("Load an image to compare rows against FFT bins.")

        self.sld_denoise_pct.setEnabled(st.denoise_db > 0)
        self.cmb_bitrate.setEnabled(st.output_format == "mp3")
        self.cmb_depth.setEnabled(st.output_format in ("wav", "flac"))
        self.sld_target.setEnabled(st.normalize)

    # -- actions -----------------------------------------------------------

    def log(self, text: str, kind: str = "info") -> None:
        colour = {"info": "#cccccc", "warn": "#e0a33e", "error": "#ff6b6b",
                  "good": "#5fd07a"}.get(kind, "#cccccc")
        self.log_view.append(f'<span style="color:{colour}">{text}</span>')
        self.status.showMessage(text.split("\n")[0][:160])

    def open_image(self) -> None:
        start = str(self.image_path.parent) if self.image_path else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Open spectrogram image", start, IMAGE_FILTER)
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        try:
            self.raw_rgb = core.load_image(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "Could not open image", str(exc))
            return
        # Orient once, here, so that what is displayed, what gets cropped and
        # what the region selects are all the same picture.
        self.rgb = core.apply_orientation(self.raw_rgb, self.current_settings())
        self.image_path = path
        h, w, _ = self.rgb.shape
        self.image_view.set_array(self.rgb)
        self.lbl_file.setText(f"{path.name}  ({w} × {h})")
        # Crop can never exceed a quarter of the image, so the sliders are
        # rescaled to the image that is actually loaded.
        for sld in (self.sld_crop_l, self.sld_crop_r):
            sld.set_limits(0, max(1, w // 4))
        for sld in (self.sld_crop_t, self.sld_crop_b):
            sld.set_limits(0, max(1, h // 4))
        self.btn_convert.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.image_view.set_selection(None)
        self.log(f"Loaded {path.name} — {w} × {h} pixels. "
                 f"Reading it as: {self._current_scheme}.", "good")
        self._update_readouts()

    def dragEnterEvent(self, event) -> None:      # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:           # noqa: N802
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_file():
                self.load_image(p)
                break

    def reset_defaults(self) -> None:
        self.apply_settings(Settings())
        self.log("Settings reset to the Audacity dialog defaults.", "good")

    def _on_hover(self, x: float, y: float) -> None:
        if self.rgb is None:
            return
        st = self.current_settings()
        h, w, _ = self.rgb.shape
        # y comes in top-down; the frequency axis runs bottom-up.
        unit = 1.0 - y
        lo, hi = core._scale_forward(np.array([st.min_freq, st.max_freq]), st.freq_scale)
        target = lo + unit * (hi - lo)
        grid = np.linspace(max(st.min_freq, 1e-3), st.max_freq, 4096)
        freq = float(np.interp(target, core._scale_forward(grid, st.freq_scale), grid))

        width = w - st.crop_left - st.crop_right
        frames = core.infer_frame_count(max(1, width), st)
        t = x * core.frames_to_duration(frames, st)
        px = self.rgb[min(h - 1, int(y * h)), min(w - 1, int(x * w))]
        lvl = float(colormap.level_from_scheme(px.reshape(1, 1, 3), st.color_scheme)[0, 0])
        if st.flip_polarity:
            lvl = 1.0 - lvl
        db = lvl * st.range_db - st.range_db - st.gain_db
        self.lbl_hover.setText(
            f"{freq:8.0f} Hz    t = {t:6.2f} s    level {lvl:.2f}"
            f"    ≈ {db:6.1f} dBFS")

    # -- conversion --------------------------------------------------------

    def start_convert(self) -> None:
        if self.rgb is None or self._thread is not None:
            return
        self._launch(self.current_settings(), preview=False)

    def _launch(self, st: Settings, preview: bool) -> None:
        """Shared by Convert and Preview - they differ only in settings."""
        if self.rgb is None or self._thread is not None:
            return
        # Whatever is playing belongs to the result about to be replaced, so
        # stop it now rather than letting it run underneath the new render.
        self.stop_playback()
        level = core.to_level(self.rgb, st)
        self._previewing = preview
        self.log(f"{'Preview' if preview else 'Converting'}: {st.color_scheme}, "
                 f"{st.freq_scale} {st.min_freq:.0f}–{st.max_freq:.0f} Hz, "
                 f"{st.window_size}×{st.zero_padding} FFT, {st.gl_iterations} iterations, "
                 f"{st.phase_init}, {level.shape[1]} columns.")

        self._thread = QThread(self)
        self._worker = ConvertWorker(level, st)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self.btn_convert.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._thread.start()

    def start_preview(self) -> None:
        """Convert only the selected seconds, at draft quality."""
        if self.rgb is None or self._thread is not None:
            return
        st = self.current_settings()
        if self.image_view.selection() is None:
            # Nothing selected: take a few seconds from the start, chosen so the
            # preview stays quick whatever the image's length.
            whole = st.copy()
            whole.preview_start, whole.preview_end = 0.0, 1.0
            full = core.frames_to_duration(
                core.infer_frame_count(self.rgb.shape[1], whole), whole)
            st.preview_start = 0.0
            st.preview_end = min(1.0, 4.0 / max(full, 0.1))
        st.quality = "Draft - fastest preview"
        for key, value in QUALITY_PRESETS[st.quality].items():
            setattr(st, key, value)
        self._launch(st, preview=True)

    def _on_orientation_changed(self, _name: str) -> None:
        """Re-orient the loaded image so the preview shows what will be read."""
        if getattr(self, "raw_rgb", None) is None:
            self._update_readouts()
            return
        self.rgb = core.apply_orientation(self.raw_rgb, self.current_settings())
        h, w, _ = self.rgb.shape
        self.image_view.set_array(self.rgb)
        self.image_view.set_selection(None)
        for sld in (self.sld_crop_l, self.sld_crop_r):
            sld.set_limits(0, max(1, w // 4))
        for sld in (self.sld_crop_t, self.sld_crop_b):
            sld.set_limits(0, max(1, h // 4))
        if self.image_path is not None:
            self.lbl_file.setText(f"{self.image_path.name}  ({w} × {h})")
        self.log(f"Layout set to {self.cmb_orientation.currentText()} — "
                 f"now reading {w} × {h}.")
        self._update_readouts()

    def _on_region(self, start: float, end: float) -> None:
        self._update_readouts()
        if self.image_view.selection() is None:
            self.log("Region cleared - Convert will use the whole image.")
        else:
            st = self.current_settings()
            width = self.rgb.shape[1] if self.rgb is not None else 0
            secs = core.frames_to_duration(
                core.infer_frame_count(max(4, int(width * (end - start))), st), st)
            self.log(f"Region {start * 100:.0f}%-{end * 100:.0f}% selected "
                     f"(about {secs:.1f} s). Convert and Preview both use it.")

    def cancel_convert(self) -> None:
        if self._worker:
            self._worker.cancel()

    def _on_progress(self, frac: float, msg: str) -> None:
        self.progress.setValue(int(frac * 1000))
        self.progress.setFormat(f"%p%  —  {msg}")

    def _teardown_thread(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.btn_convert.setEnabled(self.rgb is not None)
        self.btn_preview.setEnabled(self.rgb is not None)
        self.btn_cancel.setEnabled(False)

    def _on_finished(self, result: core.Result) -> None:
        self.result = result
        self._teardown_thread()
        self.waveform.set_audio(result.audio)
        self.btn_play.setEnabled(self.player.available)
        self.btn_export.setEnabled(True)
        preview = getattr(self, "_previewing", False)
        self.log(f"{'Preview' if preview else 'Done'} — {result.duration_s:.2f} s, "
                 f"{result.n_frames} frames, peak {result.peak_dbfs:.1f} dBFS.", "good")
        self.progress.setFormat("%p%  —  done")
        # A preview exists to be heard, so play it as soon as it is ready.
        if preview and self.player.available:
            self.start_playback()

    def _on_failed(self, message: str) -> None:
        self._teardown_thread()
        self.progress.setValue(0)
        self.progress.setFormat("%p%  —  idle")
        if message.strip() == "Cancelled.":
            self.log("Cancelled.", "warn")
        else:
            self.log(f"Conversion failed:\n<pre>{message}</pre>", "error")
            self.tabs_bottom.setCurrentWidget(self.log_view)

    # -- playback and export ----------------------------------------------

    def toggle_play(self) -> None:
        """The Play/Stop button."""
        if self.player.is_playing():
            self.stop_playback()
        else:
            self.start_playback()

    def stop_playback(self) -> None:
        """Silence whatever is playing and put the button back."""
        self.player.stop()
        self._playtimer.stop()
        self.waveform.set_playhead(None)
        self.btn_play.setText("Play")

    def start_playback(self) -> None:
        """Play the current result from the beginning.

        Deliberately not a toggle.  This is also what runs when a preview
        finishes, and a toggle there did the opposite of what it looked like:
        if the previous preview was still playing, the new one arriving would
        toggle playback OFF, so the old audio cut out at the exact moment the
        new one should have started and the new one was never heard.
        """
        if self.result is None:
            return
        self.player.stop()          # never let two results overlap
        try:
            self.player.play(self.result.audio, self.result.sample_rate)
        except Exception as exc:
            self.log(f"Playback failed: {exc}", "error")
            self.stop_playback()
            return
        self._play_pos = 0.0
        self._playtimer.start()
        self.btn_play.setText("Stop")

    def _tick_playhead(self) -> None:
        if self.result is None:
            return
        self._play_pos += self._playtimer.interval() / 1000.0
        frac = self._play_pos / max(0.001, self.result.duration_s)
        if frac >= 1.0 or not self.player.is_playing():
            self._playtimer.stop()
            self.waveform.set_playhead(None)
            self.btn_play.setText("Play")
            return
        self.waveform.set_playhead(frac)

    def export_audio(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "Nothing to export", "Convert an image first.")
            return
        st = self.current_settings()
        stem = self.image_path.stem if self.image_path else "output"
        default = str((self.image_path.parent if self.image_path else Path.cwd())
                      / f"{stem}.{st.output_format}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export audio", default,
            f"{st.output_format.upper()} (*.{st.output_format});;All files (*)")
        if not path:
            return
        try:
            written = audio_io.export(self.result.audio, self.result.sample_rate, path,
                                      st.output_format, st.mp3_bitrate_kbps, st.bit_depth)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            self.log(f"Export failed: {exc}", "error")
            return
        self.log(f"Exported {written} "
                 f"({written.stat().st_size / 1024:.0f} kB).", "good")

    # -- batch -------------------------------------------------------------

    def run_batch(self) -> None:
        in_dir = Path(self.batch_in.text().strip())
        out_dir = Path(self.batch_out.text().strip() or (in_dir / "audio"))
        if not in_dir.is_dir():
            QMessageBox.warning(self, "Batch", "Pick a valid input folder first.")
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in exts)
        if not images:
            QMessageBox.information(self, "Batch", "No images found in that folder.")
            return

        base = self.current_settings()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.tabs_bottom.setCurrentWidget(self.log_view)
        self.log(f"Batch: {len(images)} images → {out_dir}")
        self.btn_batch.setEnabled(False)
        ok = 0
        for i, img_path in enumerate(images, 1):
            QApplication.processEvents()
            try:
                rgb = core.load_image(str(img_path))
                st = base.copy()
                self.progress.setFormat(f"%p%  —  [{i}/{len(images)}] {img_path.name}")

                def report(frac: float, msg: str, i=i) -> None:
                    self.progress.setValue(int(1000 * (i - 1 + frac) / len(images)))
                    QApplication.processEvents()

                res = core.convert(core.to_level(rgb, st), st, report)
                dest = out_dir / f"{img_path.stem}.{st.output_format}"
                audio_io.export(res.audio, res.sample_rate, dest,
                                st.output_format, st.mp3_bitrate_kbps, st.bit_depth)
                self.log(f"  [{i}/{len(images)}] {img_path.name} → {dest.name} "
                         f"({res.duration_s:.1f} s)", "good")
                ok += 1
            except Exception as exc:
                self.log(f"  [{i}/{len(images)}] {img_path.name} failed: {exc}", "error")
        self.btn_batch.setEnabled(True)
        self.progress.setFormat("%p%  —  batch done")
        self.log(f"Batch finished: {ok}/{len(images)} converted.", "good")

    # -- misc --------------------------------------------------------------

    def show_about(self) -> None:
        QMessageBox.information(self, "How this works", ABOUT_TEXT)

    def closeEvent(self, event) -> None:          # noqa: N802
        self.cancel_convert()
        self._teardown_thread()
        self.player.stop()
        super().closeEvent(event)


ABOUT_TEXT = """\
Spectrogram to Audio

A spectrogram picture records how loud each frequency was at each moment, but \
it throws away phase — the timing detail that says where each wave started. \
Reconstruction therefore has two halves:

1. Undo the drawing. Brightness is turned back into a magnitude using the same \
formula the exporting program used. For Audacity's Gain/Range controls that is
    dB = level × Range − Range − Gain,
so with Gain 20 and Range 80 a white pixel is −100 dBFS and a black one −20 dBFS.

2. Invent the phase. Fast Griffin-Lim repeatedly transforms back and forth, \
keeping the magnitudes you supplied and letting the phase settle into something \
self-consistent. It never fully converges, which is why output has a faint \
metallic or watery character no matter how many iterations you use.

Two things a picture can never tell you: its sample rate and its duration. \
Pixel width comes from the zoom level it was drawn at, not from the hop size. \
Type the duration if you know it; otherwise start from the guess and use \
Time stretch by ear.
"""


def main() -> int:
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName("Spectrogram to Audio")
    win = MainWindow()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        win.load_image(Path(sys.argv[1]))
    win.show()
    return app.exec()
