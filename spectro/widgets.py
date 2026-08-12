"""Small reusable Qt widgets: a slider bonded to a number box, a collapsible
group, an image view and a waveform view."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QSizePolicy, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)


class SliderSpin(QWidget):
    """A slider and a spin box showing the same value, plus a hint label.

    Sliders alone make exact values impossible to type and spin boxes alone
    make sweeping by ear impossible, so every tunable gets both.
    """

    valueChanged = pyqtSignal(float)

    def __init__(self, minimum: float, maximum: float, value: float,
                 step: float = 1.0, decimals: int = 0, suffix: str = "",
                 tooltip: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._min, self._max, self._step = float(minimum), float(maximum), float(step)
        self._decimals = decimals
        self._guard = False

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._steps())
        # Ignored, not Expanding: the slider must be willing to shrink so the
        # panel never demands a horizontal scrollbar and clip the number boxes.
        self.slider.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.slider.setMinimumWidth(60)

        # QSpinBox is strictly integer: handing it a float raises TypeError, so
        # every value crossing this boundary is cast rather than merely rounded.
        self.spin = QDoubleSpinBox() if decimals else QSpinBox()
        if decimals:
            self.spin.setRange(float(minimum), float(maximum))
            self.spin.setDecimals(decimals)
            self.spin.setSingleStep(float(step))
        else:
            self.spin.setRange(int(round(minimum)), int(round(maximum)))
            self.spin.setSingleStep(max(1, int(round(step))))
        self.spin.setSuffix(suffix)
        self.spin.setMinimumWidth(88)
        self.spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.spin.setButtonSymbols(self.spin.ButtonSymbols.UpDownArrows)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin, 0)

        if tooltip:
            self.setToolTip(tooltip)
            self.slider.setToolTip(tooltip)
            self.spin.setToolTip(tooltip)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.set_value(value)

    def _steps(self) -> int:
        return max(1, int(round((self._max - self._min) / self._step)))

    def _spin_value(self, val: float) -> float | int:
        """Cast to whatever type this spin box accepts."""
        return float(val) if self._decimals else int(round(val))

    def _from_slider(self, pos: int) -> None:
        if self._guard:
            return
        self._guard = True
        val = self._min + pos * self._step
        self.spin.setValue(self._spin_value(val))
        self._guard = False
        self.valueChanged.emit(self.value())

    def _from_spin(self, val: float) -> None:
        if self._guard:
            return
        self._guard = True
        self.slider.setValue(int(round((float(val) - self._min) / self._step)))
        self._guard = False
        self.valueChanged.emit(self.value())

    def value(self) -> float:
        v = float(self.spin.value())
        return v if self._decimals else round(v)

    def set_value(self, val: float) -> None:
        self._guard = True
        val = max(self._min, min(self._max, float(val)))
        self.spin.setValue(self._spin_value(val))
        self.slider.setValue(int(round((val - self._min) / self._step)))
        self._guard = False

    def set_limits(self, minimum: float, maximum: float) -> None:
        """Re-range both halves at once, keeping them in agreement."""
        current = self.value()
        self._min, self._max = float(minimum), float(maximum)
        if self._decimals:
            self.spin.setRange(float(minimum), float(maximum))
        else:
            self.spin.setRange(int(round(minimum)), int(round(maximum)))
        self.slider.setRange(0, self._steps())
        self.set_value(current)


class Section(QWidget):
    """A titled group of rows that can be folded away."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._button = QToolButton()
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setChecked(True)
        self._button.setStyleSheet("QToolButton { border: none; font-weight: 600; }")
        self._button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.ArrowType.DownArrow)
        self._button.clicked.connect(self._toggle)

        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(10, 4, 2, 8)
        self.grid.setVerticalSpacing(6)
        self.grid.setColumnStretch(1, 1)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self._button)
        lay.addWidget(line)
        lay.addWidget(self.body)
        self._row = 0

    def _toggle(self, checked: bool) -> None:
        self.body.setVisible(checked)
        self._button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def add_row(self, label: str, widget: QWidget, tooltip: str = "") -> QWidget:
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
            widget.setToolTip(widget.toolTip() or tooltip)
        self.grid.addWidget(lbl, self._row, 0)
        self.grid.addWidget(widget, self._row, 1)
        self._row += 1
        return widget

    def add_wide(self, widget: QWidget) -> QWidget:
        self.grid.addWidget(widget, self._row, 0, 1, 2)
        self._row += 1
        return widget

    def add_note(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: palette(mid); font-size: 11px;")
        return self.add_wide(lbl)


class ImageView(QLabel):
    """Shows the loaded spectrogram, with zoom and pan.

    Left-drag marks a time region for the preview.  The wheel zooms about the
    pointer, middle-drag (or shift with the left button) slides the view, and a
    double-click returns to fit.  A spectrogram of anything long is thousands of
    pixels wide and a few hundred tall, so being able to magnify a couple of
    seconds and slide along is the difference between seeing syllables and
    seeing a smear.

    The picture is painted here rather than handed to QLabel, which would
    otherwise scale a pixmap to its own size and then report that size as what
    it wants to be - a loop that grew the window every time a wide image was
    loaded, and never let it shrink back.
    """

    hovered = pyqtSignal(float, float)          # normalised x, y (0..1)
    regionChanged = pyqtSignal(float, float)    # normalised start, end
    viewChanged = pyqtSignal()

    MAX_ZOOM = 60.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #1b1b1b; border: 1px solid #3a3a3a;")
        self.setText("No image loaded\n\nOpen an image, or drag one here")
        # Constant hints: see the class docstring.  The view takes whatever the
        # layout offers and never asks for more.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self._pixmap: QPixmap | None = None
        self._zoom = 1.0                       # 1.0 = whole image visible
        self._cx, self._cy = 0.5, 0.5          # image point at the view centre
        self._sel: tuple[float, float] | None = None
        self._dragging = False
        self._drag_from = 0.0
        self._panning = False
        self._pan_anchor = (0, 0)
        self._pan_start = (0.5, 0.5)

    # -- geometry ----------------------------------------------------------

    def sizeHint(self) -> QSize:            # noqa: N802
        return QSize(420, 240)

    def minimumSizeHint(self) -> QSize:     # noqa: N802
        return QSize(160, 120)

    def _scale(self) -> float:
        """Pixels on screen per pixel of image."""
        if self._pixmap is None:
            return 1.0
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if iw <= 0 or ih <= 0:
            return 1.0
        fit = min(self.width() / iw, self.height() / ih)
        return fit * self._zoom

    def _clamp(self) -> None:
        """Keep the view over the image, and centred on any axis that fits."""
        if self._pixmap is None:
            return
        scale = self._scale()
        half_w = self.width() / (2 * scale * self._pixmap.width())
        half_h = self.height() / (2 * scale * self._pixmap.height())
        self._cx = 0.5 if half_w >= 0.5 else min(max(self._cx, half_w), 1 - half_w)
        self._cy = 0.5 if half_h >= 0.5 else min(max(self._cy, half_h), 1 - half_h)

    def _origin(self) -> tuple[float, float]:
        """Where image pixel (0, 0) lands in widget coordinates."""
        if self._pixmap is None:
            return 0.0, 0.0
        scale = self._scale()
        return (self.width() / 2 - self._cx * self._pixmap.width() * scale,
                self.height() / 2 - self._cy * self._pixmap.height() * scale)

    def to_unit(self, px: float, py: float) -> tuple[float, float]:
        """Widget point -> normalised image coordinates."""
        if self._pixmap is None:
            return 0.0, 0.0
        scale = self._scale()
        ox, oy = self._origin()
        return ((px - ox) / max(1e-6, self._pixmap.width() * scale),
                (py - oy) / max(1e-6, self._pixmap.height() * scale))

    def _unit_to_x(self, u: float) -> float:
        ox = self._origin()[0]
        return ox + u * self._pixmap.width() * self._scale()

    # -- content -----------------------------------------------------------

    def set_array(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self._pixmap = None
            self.setText("No image loaded\n\nOpen an image, or drag one here")
            self.update()
            return
        arr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(img)
        self.setText("")
        self.reset_view()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._cx = self._cy = 0.5
        self.update()
        self.viewChanged.emit()

    def zoom_level(self) -> float:
        return self._zoom

    def zoom_by(self, factor: float, at: tuple[float, float] | None = None) -> None:
        """Multiply the zoom, keeping the image point under `at` in place."""
        if self._pixmap is None:
            return
        before = self.to_unit(*at) if at else (self._cx, self._cy)
        self._zoom = float(min(self.MAX_ZOOM, max(1.0, self._zoom * factor)))
        if at:
            # Put the same image point back under the pointer.
            scale = self._scale()
            self._cx = before[0] - (at[0] - self.width() / 2) / (
                self._pixmap.width() * scale)
            self._cy = before[1] - (at[1] - self.height() / 2) / (
                self._pixmap.height() * scale)
        self._clamp()
        self.update()
        self.viewChanged.emit()

    # -- region selection --------------------------------------------------

    def selection(self) -> tuple[float, float] | None:
        return self._sel

    def set_selection(self, sel: tuple[float, float] | None) -> None:
        self._sel = sel
        self.update()

    # -- events ------------------------------------------------------------

    def wheelEvent(self, event) -> None:        # noqa: N802
        if self._pixmap is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps:
            pos = event.position()
            self.zoom_by(1.25 ** steps, (pos.x(), pos.y()))
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:   # noqa: N802
        self.reset_view()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:      # noqa: N802
        if self._pixmap is not None:
            shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            if event.button() == Qt.MouseButton.MiddleButton or (
                    event.button() == Qt.MouseButton.LeftButton and shift):
                self._panning = True
                self._pan_anchor = (event.position().x(), event.position().y())
                self._pan_start = (self._cx, self._cy)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif event.button() == Qt.MouseButton.LeftButton:
                self._dragging = True
                self._drag_from = float(np.clip(
                    self.to_unit(event.position().x(), 0)[0], 0.0, 1.0))
                self._sel = None
                self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:    # noqa: N802
        if self._panning:
            self._panning = False
            self.unsetCursor()
        elif self._dragging:
            self._dragging = False
            x = float(np.clip(self.to_unit(event.position().x(), 0)[0], 0.0, 1.0))
            lo, hi = sorted((self._drag_from, x))
            # A click rather than a drag clears the region instead of making a
            # sliver that would convert to nothing.
            self._sel = (lo, hi) if hi - lo > 0.005 else None
            self.regionChanged.emit(*(self._sel or (0.0, 1.0)))
            self.update()
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:       # noqa: N802
        if self._pixmap is not None:
            px, py = event.position().x(), event.position().y()
            if self._panning:
                scale = self._scale()
                self._cx = self._pan_start[0] - (px - self._pan_anchor[0]) / (
                    self._pixmap.width() * scale)
                self._cy = self._pan_start[1] - (py - self._pan_anchor[1]) / (
                    self._pixmap.height() * scale)
                self._clamp()
                self.update()
                self.viewChanged.emit()
            else:
                u, v = self.to_unit(px, py)
                if 0 <= u <= 1 and 0 <= v <= 1:
                    self.hovered.emit(u, v)
                if self._dragging:
                    lo, hi = sorted((self._drag_from, float(np.clip(u, 0.0, 1.0))))
                    self._sel = (lo, hi)
                    self.update()
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:           # noqa: N802
        if self._pixmap is None:
            super().paintEvent(event)
            return
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1b1b1b"))
        scale = self._scale()
        ox, oy = self._origin()
        target = QRectF(ox, oy, self._pixmap.width() * scale,
                        self._pixmap.height() * scale)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale < 1.0)
        p.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        if self._sel is not None:
            x0, x1 = self._unit_to_x(self._sel[0]), self._unit_to_x(self._sel[1])
            rect = QRectF(x0, oy, x1 - x0, self._pixmap.height() * scale)
            p.fillRect(rect, QColor(80, 160, 255, 60))
            p.setPen(QPen(QColor("#4da3ff"), 1))
            p.drawRect(rect)

        if self._zoom > 1.001:
            p.setPen(QColor("#9fd0ff"))
            p.drawText(8, 18, f"{self._zoom:.1f}x   "
                              f"(wheel zooms, middle-drag or shift-drag pans, "
                              f"double-click fits)")


class WaveformView(QWidget):
    """Min/max envelope of the reconstructed audio."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._audio: np.ndarray | None = None
        self._playhead: float | None = None

    def set_audio(self, audio: np.ndarray | None) -> None:
        self._audio = audio
        self._playhead = None
        self.update()

    def set_playhead(self, frac: float | None) -> None:
        self._playhead = frac
        self.update()

    def paintEvent(self, event) -> None:        # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1b1b1b"))
        w, h = self.width(), self.height()
        mid = h / 2

        p.setPen(QPen(QColor("#3a3a3a"), 1))
        p.drawLine(0, int(mid), w, int(mid))

        if self._audio is None or len(self._audio) == 0:
            p.setPen(QColor("#777"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Waveform appears here after conversion")
            return

        # One column per pixel, drawn as the min..max span of that slice.
        n = len(self._audio)
        edges = np.linspace(0, n, w + 1).astype(np.int64)
        p.setPen(QPen(QColor("#4da3ff"), 1))
        for x in range(w):
            a, b = edges[x], max(edges[x] + 1, edges[x + 1])
            chunk = self._audio[a:b]
            lo, hi = float(chunk.min()), float(chunk.max())
            p.drawLine(x, int(mid - hi * mid * 0.95), x, int(mid - lo * mid * 0.95))

        if self._playhead is not None:
            x = int(self._playhead * w)
            p.setPen(QPen(QColor("#ff5b5b"), 1))
            p.drawLine(x, 0, x, h)
