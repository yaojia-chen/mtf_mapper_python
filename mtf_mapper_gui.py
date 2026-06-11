#!/usr/bin/env python3
"""Tkinter GUI for the standalone Python MTF Mapper port."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
import tkinter as tk

import numpy as np

import mtf_mapper_py


LOGGER = logging.getLogger("mtf_mapper_gui")
SUPPORTED_IMAGE_TYPES = [
    ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.webp *.jp2 *.j2k *.bmp *.ppm *.pgm"),
    ("Raw files", "*.raw *.bin *.dat"),
    ("All files", "*.*"),
]
RAW_FILE_SUFFIXES = {".raw", ".bin", ".dat"}
THRESHOLD_MODE_VALUES = {
    "Hybrid (adaptive + global)": "hybrid",
    "Adaptive only": "adaptive",
    "Global only": "global",
}
ESF_METHOD_VALUES = {
    "Pixel binning": "pixel-binned",
    "Auto fallback": "auto",
    "Interpolated profiles": "interpolated",
}
RAW_NORMALIZATION_VALUES = {
    "Auto levels": "auto",
    "Bit depth": "bit-depth",
    "Manual levels": "manual",
    "Full dtype range": "dtype-range",
}
PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_CHART = PROJECT_DIR / "samples" / "mtf_test_chart.png"


@dataclass
class GuiRunResult:
    input_path: Path
    output_dir: Path
    edge_count: int
    measurements: list[mtf_mapper_py.EdgeMeasurement]
    original_preview_path: Path
    annotated_preview_path: Path
    heatmap_preview_path: Path | None
    detection_report: mtf_mapper_py.DetectionReport


@dataclass
class GuiRunError:
    input_path: Path
    original_preview_path: Path | None
    error: Exception


@dataclass
class PreviewState:
    path: Path
    measurements: list[mtf_mapper_py.EdgeMeasurement]
    display_scale: float
    image_width: int
    image_height: int
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class CurvePlotState:
    values: np.ndarray
    x0: float
    y0: float
    x1: float
    y1: float
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    x_unit: str


def summarize_measurements(measurements: list[mtf_mapper_py.EdgeMeasurement]) -> dict[str, float | int]:
    if not measurements:
        return {"edges": 0, "blocks": 0, "median": 0.0, "minimum": 0.0, "maximum": 0.0}
    values = np.array([measurement.mtf_value for measurement in measurements], dtype=float)
    return {
        "edges": len(measurements),
        "blocks": len({measurement.block_id for measurement in measurements}),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def is_raw_input_path(path: Path) -> bool:
    return path.suffix.lower() in RAW_FILE_SUFFIXES


def raw_import_error_message(path: Path, error: Exception) -> str:
    return (
        f"Could not open {path.name} using the current Raw import settings.\n\n"
        "Check Read as raw pixel stream, Width, Height, Data type, Byte order, "
        "Header bytes, Channels, Channel order, and Levels in Advanced > Raw import, "
        "then click Reload with Raw settings before running analysis again. "
        "Packed 10/12/14-bit streams must be unpacked before import.\n\n"
        f"Details: {error}"
    )


def normalize_threshold_mode(value: object) -> str:
    text = str(value)
    return THRESHOLD_MODE_VALUES.get(text, text)


def normalize_esf_method(value: object) -> str:
    text = str(value)
    return ESF_METHOD_VALUES.get(text, text)


def normalize_raw_normalization(value: object) -> str:
    text = str(value)
    return RAW_NORMALIZATION_VALUES.get(text, text)


def namespace_from_gui_values(input_path: Path, output_dir: Path, values: dict[str, object]) -> argparse.Namespace:
    raw_enabled = bool(values.get("raw", False))
    args = argparse.Namespace(
        input_image=str(input_path),
        output_dir=str(output_dir),
        threshold=float(values.get("threshold", 0.55)),
        threshold_mode=normalize_threshold_mode(values.get("threshold_mode", "hybrid")),
        threshold_window=float(values.get("threshold_window", 1.0 / 3.0)),
        roi_radius=float(values.get("roi_radius", 12.0)),
        esf_method=normalize_esf_method(values.get("esf_method", "pixel-binned")),
        linear=bool(values.get("linear", False)),
        invert=bool(values.get("invert", False)),
        single_roi=bool(values.get("single_roi", False)),
        mtf_metric=str(values.get("mtf_metric", "mtf_ny4")),
        mtf=float(values.get("mtf", 50.0)),
        annotate=bool(values.get("annotate", True)),
        edges=bool(values.get("edges", True)),
        heatmap=bool(values.get("heatmap", False)),
        full_sfr=bool(values.get("full_sfr", False)),
        nosmoothing=bool(values.get("nosmoothing", False)),
        pixelsize=values.get("pixelsize"),
        raw=raw_enabled,
        raw_width=values.get("raw_width"),
        raw_height=values.get("raw_height"),
        raw_dtype=str(values.get("raw_dtype", "uint16")),
        raw_byte_order=str(values.get("raw_byte_order", "little")),
        raw_header=int(values.get("raw_header", 0)),
        raw_channels=int(values.get("raw_channels", 1)),
        raw_channel_order=str(values.get("raw_channel_order", "rgb")),
        raw_normalization=normalize_raw_normalization(values.get("raw_normalization", "auto")),
        raw_bit_depth=int(values.get("raw_bit_depth", 16)),
        raw_alignment=str(values.get("raw_alignment", "right")),
        raw_black_level=values.get("raw_black_level"),
        raw_white_level=values.get("raw_white_level"),
        log_level=str(values.get("log_level", "INFO")),
        auto_tune=bool(values.get("auto_tune", False)),
        annotation_labels=str(values.get("annotation_labels", "All values")),
        exclude_small_fiducials=bool(values.get("exclude_small_fiducials", False)),
        fiducial_max_area_ratio=float(values.get("fiducial_max_area_percent", 20.0)) / 100.0,
        manual_boxes=values.get("manual_boxes"),
        excluded_blocks=values.get("excluded_blocks", []),
    )
    if args.pixelsize in ("", None):
        args.pixelsize = None
    else:
        args.pixelsize = float(args.pixelsize)
    if raw_enabled:
        args.raw_width = int(args.raw_width)
        args.raw_height = int(args.raw_height)
        args.raw_black_level = None if args.raw_black_level in ("", None) else float(args.raw_black_level)
        args.raw_white_level = None if args.raw_white_level in ("", None) else float(args.raw_white_level)
    return args


def open_with_system(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def nearest_measurement(
    measurements: list[mtf_mapper_py.EdgeMeasurement],
    image_x: float,
    image_y: float,
) -> mtf_mapper_py.EdgeMeasurement | None:
    if not measurements:
        return None
    return min(measurements, key=lambda m: distance_to_measurement(m, image_x, image_y))


def distance_to_segment(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / denom))
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def distance_to_measurement(measurement: mtf_mapper_py.EdgeMeasurement, image_x: float, image_y: float) -> float:
    return distance_to_segment(
        image_x,
        image_y,
        measurement.edge_start_x,
        measurement.edge_start_y,
        measurement.edge_end_x,
        measurement.edge_end_y,
    )


def preview_to_image_coords(event_x: int, event_y: int, state: PreviewState) -> tuple[float, float]:
    return (event_x - state.offset_x) / state.display_scale, (event_y - state.offset_y) / state.display_scale


def wheel_zoom_factor(delta: float) -> float:
    if abs(delta) >= 120:
        return 1.15 ** (delta / 120.0)
    return math.exp(max(-4.0, min(4.0, delta)) * 0.04)


def magnify_zoom_factor(delta: float) -> float:
    return math.exp(max(-0.7, min(0.7, delta)))


def short_path(path: Path, max_chars: int = 32) -> str:
    text = str(path)
    if len(text) <= max_chars:
        return text
    return "..." + text[-(max_chars - 3) :]


def image_size_from_cv(path: Path) -> tuple[int, int]:
    mtf_mapper_py.require_cv2()
    cv2 = mtf_mapper_py.cv2
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot preview {path}")
    height, width = image.shape[:2]
    return width, height


def rgb_image_from_cv(path: Path) -> np.ndarray:
    mtf_mapper_py.require_cv2()
    cv2 = mtf_mapper_py.cv2
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot preview {path}")
    image8 = mtf_mapper_py.display_copy(image)
    if image8.ndim == 2:
        rgb = cv2.cvtColor(image8, cv2.COLOR_GRAY2RGB)
    elif image8.shape[2] == 4:
        rgb = cv2.cvtColor(image8, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image8[:, :, :3], cv2.COLOR_BGR2RGB)
    return rgb


def photo_image_from_rgb(rgb: np.ndarray, display_scale: float) -> tuple[tk.PhotoImage, int, int, int, int]:
    mtf_mapper_py.require_cv2()
    cv2 = mtf_mapper_py.cv2
    height, width = rgb.shape[:2]
    display_width = max(1, int(round(width * display_scale)))
    display_height = max(1, int(round(height * display_scale)))
    interpolation = cv2.INTER_AREA if display_scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (display_width, display_height), interpolation=interpolation)
    ok, data = cv2.imencode(".ppm", resized)
    if not ok:
        raise ValueError("Cannot render preview image")
    return tk.PhotoImage(data=data.tobytes(), format="PPM"), width, height, display_width, display_height


def photo_image_from_cv(path: Path, display_scale: float) -> tuple[tk.PhotoImage, int, int, int, int]:
    return photo_image_from_rgb(rgb_image_from_cv(path), display_scale)


def prepare_original_preview(input_path: Path, output_dir: Path, values: dict[str, object]) -> Path:
    if not bool(values.get("raw", False)):
        return input_path
    args = namespace_from_gui_values(input_path, output_dir, values)
    lum, _original = mtf_mapper_py.load_input_luminance(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "original_preview.png"
    if not mtf_mapper_py.cv2.imwrite(str(preview_path), mtf_mapper_py.display_copy(lum)):
        raise ValueError(f"Cannot create original preview for {input_path}")
    return preview_path


def curve_data(
    measurement: mtf_mapper_py.EdgeMeasurement,
    curve_type: str,
) -> tuple[np.ndarray, float, float, str, str, str]:
    if curve_type == "ESF":
        values = measurement.esf
        half_span = max((len(values) - 1) * measurement.sample_spacing / 2.0, 0.0)
        return values, -half_span, half_span, "Distance across edge (pixels)", "ESF", "px"
    if curve_type == "LSF":
        values = measurement.lsf
        half_span = max((len(values) - 1) * measurement.sample_spacing / 2.0, 0.0)
        return values, -half_span, half_span, "Distance across edge (pixels)", "LSF", "px"
    values = measurement.sfr
    return values, 0.0, (len(values) - 1) / 64.0, "Frequency (cycles/pixel)", "SFR", "c/p"


def draw_curve(canvas: tk.Canvas, measurement: mtf_mapper_py.EdgeMeasurement, curve_type: str) -> CurvePlotState | None:
    canvas.delete("all")
    width = max(canvas.winfo_width(), 320)
    height = max(canvas.winfo_height(), 220)
    left, right, top, bottom = 68, 20, 20, 70
    plot_w = max(width - left - right, 1)
    plot_h = max(height - top - bottom, 1)
    x0, y0 = left, height - bottom
    x1, y1 = width - right, top

    values, x_min, x_max, x_label, y_label, x_unit = curve_data(measurement, curve_type)
    if len(values) < 2:
        return None
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        return None
    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if curve_type in ("SFR", "ESF"):
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 1.0)
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0

    canvas.create_rectangle(x0, y1, x1, y0, outline="#a0a0a0")
    for tick in range(5):
        frac = tick / 4
        y = y0 - frac * plot_h
        canvas.create_line(x0, y, x1, y, fill="#ececec")
        value = y_min + frac * (y_max - y_min)
        canvas.create_text(x0 - 8, y, text=f"{value:.2f}", anchor=tk.E, fill="#555555", font=("TkDefaultFont", 9))
    for tick in range(5):
        frac = tick / 4
        x = x0 + frac * plot_w
        value = x_min + frac * (x_max - x_min)
        canvas.create_line(x, y0, x, y0 + 4, fill="#666666")
        canvas.create_text(x, y0 + 18, text=f"{value:.2f}", anchor=tk.N, fill="#555555", font=("TkDefaultFont", 9))

    points: list[float] = []
    for idx, value in enumerate(values):
        x = x0 + (idx / (len(values) - 1)) * plot_w
        y = y0 - (float(value) - y_min) / (y_max - y_min) * plot_h
        points.extend([x, y])
    canvas.create_line(*points, fill="#0b63ce", width=2, smooth=True)
    if curve_type == "SFR":
        if measurement.mtf_column == "mtf50":
            guide_x = measurement.mtf_value
            guide_y = 0.5
            guide_label = f"MTF50 {measurement.mtf_value:.3f}"
        else:
            guide_x = 0.25 if measurement.mtf_column == "mtf_ny4" else 0.5
            guide_y = measurement.mtf_value
            guide_label = f"{measurement.mtf_column} {measurement.mtf_value:.3f}"
        if x_min <= guide_x <= x_max:
            x = x0 + (guide_x - x_min) / max(x_max - x_min, 1e-12) * plot_w
            canvas.create_line(x, y1, x, y0, fill="#d12b2b", dash=(5, 4))
        if y_min <= guide_y <= y_max:
            y = y0 - (guide_y - y_min) / max(y_max - y_min, 1e-12) * plot_h
            canvas.create_line(x0, y, x1, y, fill="#d12b2b", dash=(5, 4))
        canvas.create_text(x1 - 6, y1 + 6, text=guide_label, anchor=tk.NE, fill="#a12020")
        if x_min <= 0.5 <= x_max:
            nyquist_x = x0 + (0.5 - x_min) / max(x_max - x_min, 1e-12) * plot_w
            canvas.create_line(nyquist_x, y1, nyquist_x, y0, fill="#777777", dash=(2, 5))
            canvas.create_text(nyquist_x + 4, y0 - 4, text="Nyquist", anchor=tk.SW, fill="#666666")
    canvas.create_text(width // 2, height - 8, text=x_label, anchor=tk.S, fill="#333333")
    canvas.create_text(14, (y0 + y1) / 2, text=y_label, angle=90, fill="#333333")
    return CurvePlotState(values, x0, y0, x1, y1, y_min, y_max, x_min, x_max, x_unit)


class MtfMapperGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MTF Mapper Python")
        self.geometry("1280x820")
        self.minsize(1080, 680)

        self.input_files: list[Path] = []
        self.result_rows: dict[str, Path] = {}
        self.result_measurements: dict[Path, list[mtf_mapper_py.EdgeMeasurement]] = {}
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview_image: tk.PhotoImage | None = None
        self.preview_rgb: np.ndarray | None = None
        self.preview_rgb_path: Path | None = None
        self.preview_state: PreviewState | None = None
        self.preview_path: Path | None = None
        self.preview_measurements: list[mtf_mapper_py.EdgeMeasurement] = []
        self.original_preview_path: Path | None = None
        self.annotated_preview_path: Path | None = None
        self.annotated_preview_measurements: list[mtf_mapper_py.EdgeMeasurement] = []
        self.heatmap_preview_path: Path | None = None
        self.detection_preview_path: Path | None = None
        self.threshold_preview_path: Path | None = None
        self.manual_boxes: list[np.ndarray] = []
        self.excluded_blocks: set[int] = set()
        self.manual_roi_active = False
        self.detection_report: mtf_mapper_py.DetectionReport | None = None
        self.preview_zoom = 1.0
        self.preview_fit_scale = 1.0
        self.preview_drag_start: tuple[int, int] | None = None
        self.pending_zoom_factor = 1.0
        self.pending_zoom_anchor: tuple[int, int] | None = None
        self.pending_zoom_after: str | None = None
        self.pending_render_after: str | None = None
        self.selected_measurement: mtf_mapper_py.EdgeMeasurement | None = None
        self.curve_plot_state: CurvePlotState | None = None
        self.edge_inspector: tk.Toplevel | None = None
        self.sfr_canvas: tk.Canvas | None = None
        self.raw_widgets: list[tk.Widget] = []
        self.raw_bit_widgets: list[tk.Widget] = []
        self.raw_manual_widgets: list[tk.Widget] = []
        self.current_output_root = Path(tempfile.gettempdir()) / "mtf_mapper_python_gui"
        self.dock_collapsed = False
        self.selected_detection_block: int | None = None

        self._build_vars()
        self._configure_style()
        self._build_menu()
        self._build_layout()
        self._update_raw_controls()
        self.after(100, self._poll_worker_queue)

    def _build_vars(self) -> None:
        self.threshold = DoubleVar(value=0.55)
        self.threshold_mode = StringVar(value="Hybrid (adaptive + global)")
        self.threshold_window = DoubleVar(value=0.333)
        self.roi_radius = DoubleVar(value=12.0)
        self.esf_method = StringVar(value="Pixel binning")
        self.linear = BooleanVar(value=False)
        self.invert = BooleanVar(value=False)
        self.single_roi = BooleanVar(value=False)
        self.mtf_metric = StringVar(value="mtf_ny4")
        self.mtf = DoubleVar(value=50.0)
        self.annotate = BooleanVar(value=True)
        self.edges = BooleanVar(value=True)
        self.heatmap = BooleanVar(value=False)
        self.full_sfr = BooleanVar(value=False)
        self.nosmoothing = BooleanVar(value=False)
        self.auto_tune = BooleanVar(value=False)
        self.exclude_small_fiducials = BooleanVar(value=False)
        self.fiducial_max_area_percent = DoubleVar(value=20.0)
        self.quality_filter = StringVar(value="All edges")
        self.annotation_labels = StringVar(value="All values")
        self.pixelsize = StringVar(value="")
        self.raw = BooleanVar(value=False)
        self.raw_width = StringVar(value="")
        self.raw_height = StringVar(value="")
        self.raw_dtype = StringVar(value="uint16")
        self.raw_byte_order = StringVar(value="little")
        self.raw_header = IntVar(value=0)
        self.raw_channels = IntVar(value=1)
        self.raw_channel_order = StringVar(value="rgb")
        self.raw_normalization = StringVar(value="Auto levels")
        self.raw_bit_depth = IntVar(value=16)
        self.raw_alignment = StringVar(value="right")
        self.raw_black_level = StringVar(value="")
        self.raw_white_level = StringVar(value="")
        self.status = StringVar(value="Ready")
        self.workflow_hint = StringVar(value="1. Open an image or try the sample")
        self.preview_guide = StringVar(value="")
        self.diagnostics_source = StringVar(value="No diagnostics yet")
        self.summary_title = StringVar(value="No analysis yet")
        self.summary_detail = StringVar(value="Open an image or try the sample chart to see a measurement summary.")
        self.summary_edges = StringVar(value="-")
        self.summary_blocks = StringVar(value="-")
        self.summary_median = StringVar(value="-")
        self.summary_range = StringVar(value="-")
        self.preview_info = StringVar(value="No image")
        self.preview_mode = StringVar(value="Original")
        self.curve_type = StringVar(value="SFR")
        self.selected_edge = StringVar(value="No edge selected")
        self.raw.trace_add("write", lambda *_args: self._update_raw_controls())
        self.raw_normalization.trace_add("write", lambda *_args: self._update_raw_controls())
        self.preview_mode.trace_add("write", lambda *_args: self.show_preview_mode())
        self.curve_type.trace_add("write", lambda *_args: self.redraw_selected_curve())

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "aqua" in style.theme_names():
            style.theme_use("aqua")
        style.configure("Muted.TLabel", foreground="#5f6b7a")
        style.configure("SummaryTitle.TLabel", font=("TkDefaultFont", 13, "bold"))
        style.configure("StatValue.TLabel", font=("TkDefaultFont", 14, "bold"))
        style.configure("Primary.TButton", font=("TkDefaultFont", 11, "bold"))
        style.configure("Section.TLabel", font=("TkDefaultFont", 10, "bold"))
        style.configure("Guide.TLabel", foreground="#31586d")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        shortcut = "Cmd" if sys.platform == "darwin" else "Ctrl"
        file_menu.add_command(label="Open...", accelerator=f"{shortcut}+O", command=self.open_files)
        file_menu.add_command(label="Open single edge image...", accelerator=f"{shortcut}+E", command=self.open_single_roi)
        file_menu.add_separator()
        file_menu.add_command(label="Choose output directory...", command=self.choose_output_dir)
        file_menu.add_separator()
        file_menu.add_command(label="Save project...", command=self.save_project)
        file_menu.add_command(label="Open project...", command=self.open_project)
        file_menu.add_command(label="Save settings preset...", command=self.save_preset)
        file_menu.add_command(label="Load settings preset...", command=self.load_preset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", accelerator=f"{shortcut}+Q", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)
        self.bind("<Control-o>", lambda _event: self.open_files())
        self.bind("<Control-e>", lambda _event: self.open_single_roi())
        self.bind("<Control-q>", lambda _event: self.destroy())
        self.bind("<Control-r>", lambda _event: self.run_analysis())
        if sys.platform == "darwin":
            self.bind("<Command-o>", lambda _event: self.open_files())
            self.bind("<Command-e>", lambda _event: self.open_single_roi())
            self.bind("<Command-q>", lambda _event: self.destroy())
            self.bind("<Command-r>", lambda _event: self.run_analysis())

    def _build_layout(self) -> None:
        self._build_toolbar()

        workspace = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        workspace.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self.workspace = workspace
        self._build_image_panel(workspace)
        self._build_bottom_tabs(workspace)
        self.after(350, self._set_initial_workspace_split)
        status_bar = ttk.Frame(self)
        status_bar.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(status_bar, textvariable=self.status, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=140)
        self.progress.pack(side=tk.RIGHT)

    def _set_initial_workspace_split(self) -> None:
        try:
            self.workspace.sashpos(0, max(620, self.workspace.winfo_width() - 470))
            self.after(50, self.fit_preview)
        except tk.TclError:
            pass

    def show_dock_tab(self, tab: ttk.Frame) -> None:
        self.bottom_tabs.select(tab)
        if self.dock_collapsed:
            self.toggle_dock()
        else:
            self.after(50, self.fit_preview)

    def toggle_dock(self) -> None:
        self.dock_collapsed = not self.dock_collapsed
        dock_width = 42 if self.dock_collapsed else 470
        try:
            self.workspace.sashpos(0, max(560, self.workspace.winfo_width() - dock_width))
            self.after(50, self.fit_preview)
        except tk.TclError:
            pass
        self.dock_button.config(text="Show dock" if self.dock_collapsed else "Hide dock")

    def update_workflow_actions(self) -> None:
        has_input = bool(self.input_files)
        has_rois = bool(self.manual_boxes)
        self.preview_detection_button.configure(state=tk.NORMAL if has_input else tk.DISABLED)
        self.edit_rois_button.configure(state=tk.NORMAL if has_rois else tk.DISABLED)
        self.run_button.configure(state=tk.NORMAL if has_input else tk.DISABLED)
        if not has_input:
            self.workflow_hint.set("1. Open an image or try the sample")
        elif self.manual_roi_active:
            included = len(self.manual_boxes) - len(self.excluded_blocks)
            self.workflow_hint.set(f"2. Detection tuned: {included} target(s) included")
        elif self.result_measurements:
            self.workflow_hint.set("Analysis complete. Inspect an edge or tune detection")
        else:
            self.workflow_hint.set("2. Preview detection or 3. Run analysis")

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 8))
        ttk.Label(toolbar, text="1  Open", style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Open image", command=self.open_files).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Try sample", command=self.open_sample).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(toolbar, text="2  Tune", style="Section.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.preview_detection_button = ttk.Button(toolbar, text="Preview detection", command=self.preview_detection)
        self.preview_detection_button.pack(side=tk.LEFT)
        self.edit_rois_button = ttk.Button(toolbar, text="Edit ROIs", command=self.edit_rois)
        self.edit_rois_button.pack(side=tk.LEFT, padx=(6, 14))
        ttk.Button(toolbar, text="Settings", command=lambda: self.show_dock_tab(self.setup_tab)).pack(side=tk.LEFT)
        self.dock_button = ttk.Button(toolbar, text="Hide dock", command=self.toggle_dock)
        self.dock_button.pack(side=tk.LEFT, padx=(6, 0))
        more_menu = tk.Menu(toolbar, tearoff=False)
        more_menu.add_command(label="Choose output folder...", command=self.choose_output_dir)
        more_menu.add_command(label="Clear results", command=self.clear_results)
        ttk.Menubutton(toolbar, text="More", menu=more_menu).pack(side=tk.RIGHT)
        self.run_button = ttk.Button(toolbar, text="3  Run analysis", command=self.run_analysis, style="Primary.TButton")
        self.run_button.pack(side=tk.RIGHT, padx=(0, 8))
        self.update_workflow_actions()

    def _build_image_panel(self, parent: ttk.PanedWindow) -> None:
        preview_frame = ttk.LabelFrame(parent, text="Image preview")
        parent.add(preview_frame, weight=4)
        controls = ttk.Frame(preview_frame)
        controls.pack(fill=tk.X, padx=8, pady=(6, 0))
        ttk.Button(controls, text="Zoom +", command=lambda: self.schedule_center_zoom(1.25)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Zoom -", command=lambda: self.schedule_center_zoom(0.8)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(controls, text="Fit", command=self.fit_preview).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(controls, text="View").pack(side=tk.LEFT, padx=(16, 4))
        self.preview_mode_box = ttk.Combobox(
            controls,
            textvariable=self.preview_mode,
            values=("Original",),
            state="readonly",
            width=10,
        )
        self.preview_mode_box.pack(side=tk.LEFT)
        ttk.Label(controls, textvariable=self.preview_info, style="Muted.TLabel", anchor=tk.E).pack(side=tk.RIGHT)
        ttk.Label(preview_frame, textvariable=self.preview_guide, style="Guide.TLabel", anchor=tk.W).pack(
            fill=tk.X, padx=10, pady=(4, 0)
        )

        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.preview = tk.Canvas(
            canvas_frame,
            background="#dfe3e8",
            highlightthickness=1,
            highlightbackground="#aab2bd",
        )
        self.preview.grid(row=0, column=0, sticky="nsew")
        x_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.preview.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.preview.yview)
        self.preview_x_scroll = x_scroll
        self.preview_y_scroll = y_scroll
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid_remove()
        y_scroll.grid_remove()
        self.preview.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.preview.create_text(320, 200, text="Open an image to run analysis", fill="#555555", tags="placeholder")
        self.preview.bind("<ButtonPress-1>", self.on_preview_press)
        self.preview.bind("<B1-Motion>", self.on_preview_drag)
        self.preview.bind("<ButtonRelease-1>", self.on_preview_release)
        self.preview.bind("<MouseWheel>", self.on_preview_wheel)
        try:
            self.preview.bind("<Magnify>", self.on_preview_magnify)
        except tk.TclError:
            pass
        self.preview.bind("<Configure>", self.on_preview_configure)

    def _build_settings_tabs(self, parent: ttk.Notebook) -> None:
        self.setup_tabs = parent
        self.setup_tab = ttk.Frame(parent)
        self.advanced_tab = ttk.Frame(parent)
        parent.add(self.setup_tab, text="Setup")
        parent.add(self.advanced_tab, text="Advanced")
        advanced_canvas = tk.Canvas(self.advanced_tab, highlightthickness=0, yscrollincrement=20)
        advanced_scroll = ttk.Scrollbar(self.advanced_tab, orient=tk.VERTICAL, command=advanced_canvas.yview)
        advanced_canvas.configure(yscrollcommand=advanced_scroll.set)
        advanced_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        advanced_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        advanced_content = ttk.Frame(advanced_canvas)
        advanced_window = advanced_canvas.create_window((0, 0), window=advanced_content, anchor=tk.NW)
        advanced_content.bind(
            "<Configure>",
            lambda _event: advanced_canvas.configure(scrollregion=advanced_canvas.bbox("all")),
        )
        advanced_canvas.bind(
            "<Configure>",
            lambda event: advanced_canvas.itemconfigure(advanced_window, width=event.width),
        )
        advanced_canvas.bind(
            "<MouseWheel>",
            lambda event: advanced_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"),
        )

        basic_tab = self.setup_tab

        props = ttk.LabelFrame(basic_tab, text="1. Image")
        props.pack(fill=tk.X, pady=(0, 8))
        self.input_label = ttk.Label(props, text="Input: none", anchor=tk.W, wraplength=420, justify=tk.LEFT)
        self.input_label.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        self.output_label = ttk.Label(
            props,
            text=f"Output: {short_path(self.current_output_root)}",
            anchor=tk.W,
            wraplength=420,
            justify=tk.LEFT,
        )
        self.output_label.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        props.columnconfigure(0, weight=1)

        settings = ttk.LabelFrame(basic_tab, text="2. Input interpretation")
        settings.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(settings, text="Linear gamma (8 bit)", variable=self.linear).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(settings, text="Invert brightness", variable=self.invert).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(settings, text="Single edge / ROI", variable=self.single_roi).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        settings.columnconfigure(1, weight=1)

        measurement = ttk.LabelFrame(basic_tab, text="3. Measurement")
        measurement.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(measurement, text="Reported metric").grid(row=0, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Combobox(
            measurement,
            textvariable=self.mtf_metric,
            values=("mtf_ny4", "mtf_ny2", "mtf50"),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        self._labeled_entry(measurement, "Target threshold", self.threshold, 1)
        self._labeled_entry(measurement, "Edge ROI radius (px)", self.roi_radius, 2)
        self._labeled_entry(measurement, "Pixel size (um)", self.pixelsize, 3)
        self._labeled_entry(measurement, "MTF contrast (%)", self.mtf, 4)
        measurement.columnconfigure(1, weight=1)

        outputs = ttk.LabelFrame(basic_tab, text="4. Outputs")
        outputs.pack(fill=tk.X)
        ttk.Checkbutton(outputs, text="Annotated image", variable=self.annotate).grid(row=0, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(outputs, text="CSV tables", variable=self.edges).grid(row=1, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(outputs, text="MTF heat map", variable=self.heatmap).grid(row=2, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Label(outputs, text="Annotation labels").grid(row=3, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Combobox(
            outputs,
            textvariable=self.annotation_labels,
            values=("All values", "Markers only"),
            state="readonly",
            width=16,
        ).grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        outputs.columnconfigure(1, weight=1)

        advanced = ttk.LabelFrame(advanced_content, text="Detection")
        advanced.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(advanced, text="Threshold mode").grid(row=0, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Combobox(
            advanced,
            textvariable=self.threshold_mode,
            values=tuple(THRESHOLD_MODE_VALUES),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        self._labeled_entry(advanced, "Adaptive window", self.threshold_window, 1)
        ttk.Checkbutton(advanced, text="Automatically tune detection", variable=self.auto_tune).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(advanced, text="Exclude small fiducials", variable=self.exclude_small_fiducials).grid(
            row=3, column=0, sticky=tk.W, padx=8, pady=3
        )
        fiducial_limit = ttk.Frame(advanced)
        fiducial_limit.grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        ttk.Entry(fiducial_limit, textvariable=self.fiducial_max_area_percent, width=7).pack(side=tk.LEFT)
        ttk.Label(fiducial_limit, text="% of largest").pack(side=tk.LEFT, padx=(4, 0))
        advanced.columnconfigure(1, weight=1)

        sfr = ttk.LabelFrame(advanced_content, text="SFR and edge quality")
        sfr.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(sfr, text="Extended SFR domain", variable=self.full_sfr).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(sfr, text="Reduced SFR smoothing", variable=self.nosmoothing).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        ttk.Label(sfr, text="ESF construction").grid(row=2, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Combobox(
            sfr,
            textvariable=self.esf_method,
            values=tuple(ESF_METHOD_VALUES),
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        ttk.Label(sfr, text="Quality filter").grid(row=3, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Combobox(
            sfr,
            textvariable=self.quality_filter,
            values=("All edges", "Good only", "Good + Review"),
            state="readonly",
        ).grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        sfr.columnconfigure(1, weight=1)

        raw = ttk.LabelFrame(advanced_content, text="Raw import")
        raw.pack(fill=tk.X)
        raw_enable = ttk.Checkbutton(raw, text="Read as raw pixel stream", variable=self.raw)
        raw_enable.pack(anchor=tk.W, padx=8, pady=4)
        self.reload_raw_button = ttk.Button(raw, text="Reload with Raw settings", command=self.reload_original_preview)
        self.reload_raw_button.pack(anchor=tk.W, padx=8, pady=(0, 6))
        self.raw_fields = ttk.Frame(raw)
        self.raw_fields.pack(fill=tk.X)
        self._labeled_entry(self.raw_fields, "Width", self.raw_width, 0, store=self.raw_widgets)
        self._labeled_entry(self.raw_fields, "Height", self.raw_height, 1, store=self.raw_widgets)
        dtype_label = ttk.Label(self.raw_fields, text="Data type")
        dtype_label.grid(row=2, column=0, sticky=tk.W, padx=8, pady=3)
        dtype_box = ttk.Combobox(self.raw_fields, textvariable=self.raw_dtype, values=("uint8", "uint16", "int16", "float32", "float64"), state="readonly")
        dtype_box.grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        order_label = ttk.Label(self.raw_fields, text="Byte order")
        order_label.grid(row=3, column=0, sticky=tk.W, padx=8, pady=3)
        order_box = ttk.Combobox(self.raw_fields, textvariable=self.raw_byte_order, values=("little", "big", "native"), state="readonly")
        order_box.grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        self.raw_widgets.extend([dtype_label, dtype_box, order_label, order_box])
        self._labeled_entry(self.raw_fields, "Header bytes", self.raw_header, 4, store=self.raw_widgets)
        self._labeled_entry(self.raw_fields, "Channels", self.raw_channels, 5, store=self.raw_widgets)
        channel_order_label = ttk.Label(self.raw_fields, text="Channel order")
        channel_order_label.grid(row=6, column=0, sticky=tk.W, padx=8, pady=3)
        channel_order_box = ttk.Combobox(
            self.raw_fields, textvariable=self.raw_channel_order, values=("rgb", "bgr"), state="readonly"
        )
        channel_order_box.grid(row=6, column=1, sticky="ew", padx=8, pady=3)
        self.raw_widgets.extend([channel_order_label, channel_order_box])
        levels_label = ttk.Label(self.raw_fields, text="Levels")
        levels_label.grid(row=7, column=0, sticky=tk.W, padx=8, pady=3)
        levels_box = ttk.Combobox(
            self.raw_fields, textvariable=self.raw_normalization, values=tuple(RAW_NORMALIZATION_VALUES), state="readonly"
        )
        levels_box.grid(row=7, column=1, sticky="ew", padx=8, pady=3)
        self.raw_widgets.extend([levels_label, levels_box])
        depth_label = ttk.Label(self.raw_fields, text="Bit depth")
        depth_label.grid(row=8, column=0, sticky=tk.W, padx=8, pady=3)
        depth_box = ttk.Combobox(self.raw_fields, textvariable=self.raw_bit_depth, values=(8, 10, 12, 14, 16), state="readonly")
        depth_box.grid(row=8, column=1, sticky="ew", padx=8, pady=3)
        align_label = ttk.Label(self.raw_fields, text="Alignment")
        align_label.grid(row=9, column=0, sticky=tk.W, padx=8, pady=3)
        align_box = ttk.Combobox(self.raw_fields, textvariable=self.raw_alignment, values=("right", "left"), state="readonly")
        align_box.grid(row=9, column=1, sticky="ew", padx=8, pady=3)
        self.raw_widgets.extend([depth_label, depth_box, align_label, align_box])
        self.raw_bit_widgets.extend([depth_label, depth_box, align_label, align_box])
        self._labeled_entry(self.raw_fields, "Black level", self.raw_black_level, 10, store=self.raw_manual_widgets)
        self._labeled_entry(self.raw_fields, "White level", self.raw_white_level, 11, store=self.raw_manual_widgets)
        self.raw_widgets.extend(self.raw_manual_widgets)
        self.raw_fields.columnconfigure(1, weight=1)

    def _build_bottom_tabs(self, parent: ttk.PanedWindow) -> None:
        self.bottom_tabs = ttk.Notebook(parent)
        parent.add(self.bottom_tabs, weight=1)
        self._build_settings_tabs(self.bottom_tabs)

        results_tab = ttk.Frame(self.bottom_tabs)
        self.results_tab = results_tab
        summary = ttk.Frame(results_tab, padding=(10, 8))
        summary.pack(fill=tk.X)
        ttk.Label(
            summary,
            textvariable=self.summary_title,
            style="SummaryTitle.TLabel",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        ttk.Label(
            summary,
            textvariable=self.summary_detail,
            style="Muted.TLabel",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 8))
        stats = ttk.Frame(summary)
        stats.pack(fill=tk.X)
        self._summary_stat(stats, "Edges", self.summary_edges, 0)
        self._summary_stat(stats, "Blocks", self.summary_blocks, 1)
        self._summary_stat(stats, "Median MTF", self.summary_median, 2)
        self._summary_stat(stats, "Range", self.summary_range, 3)
        self.result_tree = ttk.Treeview(results_tab, columns=("path",), show="tree", height=7)
        self.result_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.result_tree.bind("<<TreeviewSelect>>", self.on_result_selected)
        result_buttons = ttk.Frame(results_tab)
        result_buttons.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(result_buttons, text="Open selected", command=self.open_selected_result).pack(side=tk.LEFT)
        ttk.Button(result_buttons, text="Open output folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=(6, 0))
        self.bottom_tabs.add(results_tab, text="Result")

        log_tab = ttk.Frame(self.bottom_tabs)
        self.log_text = ScrolledText(log_tab, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.log_text.config(state=tk.DISABLED)
        self.bottom_tabs.add(log_tab, text="Log")

        diagnostics_tab = ttk.Frame(self.bottom_tabs)
        ttk.Label(diagnostics_tab, textvariable=self.diagnostics_source, style="SummaryTitle.TLabel").pack(
            anchor=tk.W, padx=8, pady=(8, 0)
        )
        self.diagnostics_text = ScrolledText(diagnostics_tab, height=8, wrap=tk.WORD)
        self.diagnostics_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.diagnostics_text.config(state=tk.DISABLED)
        self.bottom_tabs.add(diagnostics_tab, text="Diagnostics")

    def _summary_stat(self, parent: ttk.Frame, label: str, variable: StringVar, column: int) -> None:
        card = ttk.Frame(parent, padding=(10, 4))
        card.grid(row=0, column=column, sticky="ew", padx=(0, 6))
        ttk.Label(card, text=label, style="Muted.TLabel").pack(anchor=tk.W)
        ttk.Label(card, textvariable=variable, style="StatValue.TLabel").pack(anchor=tk.W)
        parent.columnconfigure(column, weight=1)

    def _labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        row: int,
        store: list[tk.Widget] | None = None,
    ) -> None:
        label_widget = ttk.Label(parent, text=label)
        entry_widget = ttk.Entry(parent, textvariable=variable)
        label_widget.grid(row=row, column=0, sticky=tk.W, padx=8, pady=3)
        entry_widget.grid(row=row, column=1, sticky="ew", padx=8, pady=3)
        if store is not None:
            store.extend([label_widget, entry_widget])

    def open_files(self, auto_run: bool = False) -> None:
        filenames = filedialog.askopenfilenames(title="Select input files", filetypes=SUPPORTED_IMAGE_TYPES)
        if not filenames:
            return
        self.input_files = [Path(name) for name in filenames]
        self.manual_boxes = []
        self.excluded_blocks.clear()
        self.manual_roi_active = False
        self.detection_preview_path = None
        self.threshold_preview_path = None
        self.single_roi.set(False)
        self.raw.set(is_raw_input_path(self.input_files[0]))
        self._update_raw_controls()
        self.input_label.config(text=f"Input: {len(self.input_files)} file(s), {self.input_files[0].name}")
        self.reload_original_preview(show_errors=False)
        self.update_workflow_actions()
        if auto_run:
            self.after(50, self.run_analysis)

    def open_single_roi(self) -> None:
        self.open_files(auto_run=False)
        if self.input_files:
            self.single_roi.set(True)
            self.after(50, self.run_analysis)

    def open_sample(self) -> None:
        if not SAMPLE_CHART.exists():
            messagebox.showerror("Sample missing", f"Sample chart not found:\n{SAMPLE_CHART}")
            return
        self.input_files = [SAMPLE_CHART]
        self.manual_boxes = []
        self.excluded_blocks.clear()
        self.manual_roi_active = False
        self.detection_preview_path = None
        self.threshold_preview_path = None
        self.linear.set(True)
        self.single_roi.set(False)
        self.raw.set(False)
        self.input_label.config(text=f"Input: sample, {SAMPLE_CHART.name}")
        self.status.set("Sample loaded; analysis will start")
        self.log(f"Loaded sample chart: {SAMPLE_CHART}")
        self.set_preview_sources(SAMPLE_CHART, None, [])
        self.update_workflow_actions()
        self.after(50, self.run_analysis)

    def reload_original_preview(self, show_errors: bool = True) -> None:
        if not self.input_files:
            return
        input_path = self.input_files[0]
        preview_dir = Path(tempfile.gettempdir()) / "mtf_mapper_python_gui_previews" / input_path.stem
        try:
            original_path = prepare_original_preview(input_path, preview_dir, self.current_values())
            self.detection_preview_path = None
            self.threshold_preview_path = None
            self.set_preview_sources(original_path, None, [])
            self.status.set("Image reloaded with current settings; preview detection or run analysis")
            self.log(f"Reloaded image preview: {input_path.name}")
        except Exception as exc:
            self.original_preview_path = None
            self.preview_path = None
            self.preview_rgb = None
            self.preview_rgb_path = None
            self.clear_preview("Set Raw import metadata, then click Reload with Raw settings")
            self.status.set("Check Raw import settings, then reload image")
            self.show_dock_tab(self.advanced_tab)
            self.log(f"Could not reload {input_path.name}: {exc}")
            if show_errors:
                messagebox.showwarning("Check Raw import settings", raw_import_error_message(input_path, exc))

    def choose_output_dir(self) -> None:
        dirname = filedialog.askdirectory(title="Select output directory")
        if not dirname:
            return
        self.current_output_root = Path(dirname)
        self.output_label.config(text=f"Output: {short_path(self.current_output_root)}")
        self.log(f"Output folder set to: {self.current_output_root}")

    def preview_detection(self) -> None:
        if not self.input_files:
            messagebox.showwarning("No input", "Select an input image first.")
            return
        try:
            input_path = self.input_files[0]
            values = self.current_values()
            args = namespace_from_gui_values(input_path, self.current_output_root / input_path.stem, values)
            lum, original = mtf_mapper_py.load_input_luminance(args)
            if self.auto_tune.get():
                boxes, report = mtf_mapper_py.auto_tune_detection(
                    lum, args.fiducial_max_area_ratio if args.exclude_small_fiducials else 0.0
                )
                self.threshold_mode.set(next((label for label, mode in THRESHOLD_MODE_VALUES.items() if mode == report.threshold_mode), report.threshold_mode))
                self.threshold.set(report.threshold)
                self.threshold_window.set(report.threshold_window)
            else:
                boxes, report = mtf_mapper_py.detect_boxes_with_diagnostics(
                    lum,
                    args.threshold,
                    args.threshold_window,
                    args.threshold_mode,
                    args.fiducial_max_area_ratio if args.exclude_small_fiducials else 0.0,
                )
            self.manual_boxes = boxes
            self.excluded_blocks.clear()
            self.manual_roi_active = True
            self.selected_detection_block = None
            preview = mtf_mapper_py.make_detection_preview(lum, original, boxes)
            preview_dir = Path(tempfile.gettempdir()) / "mtf_mapper_python_gui_previews" / input_path.stem
            preview_dir.mkdir(parents=True, exist_ok=True)
            self.detection_preview_path = preview_dir / "detection_preview.png"
            if not mtf_mapper_py.cv2.imwrite(str(self.detection_preview_path), preview):
                raise ValueError("Could not create detection preview")
            mask = mtf_mapper_py.threshold_dark_objects(lum, report.threshold, report.threshold_window, report.threshold_mode)
            self.threshold_preview_path = preview_dir / "threshold_mask.png"
            if not mtf_mapper_py.cv2.imwrite(str(self.threshold_preview_path), mask):
                raise ValueError("Could not create threshold mask preview")
            self.detection_report = report
            self.set_diagnostics(report, [], source="Detection preview")
            self.set_preview_sources(self.original_preview_path or input_path, self.annotated_preview_path, self.annotated_preview_measurements, "Detection", self.heatmap_preview_path)
            self.status.set("Detection preview: click a target to include/exclude it; Shift-drag to add an ROI")
            self.update_workflow_actions()
        except Exception as exc:
            messagebox.showerror("Detection preview failed", str(exc))

    def edit_rois(self) -> None:
        if not self.manual_boxes:
            messagebox.showinfo("No ROIs", "Run Preview detection first, or Shift-drag in the detection view to add an ROI.")
            return
        self.preview_mode.set("Detection")
        self.selected_detection_block = 1
        self.refresh_detection_preview()
        dialog = tk.Toplevel(self)
        dialog.title("Edit target ROI")
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        roi_number = IntVar(value=1)
        x_value = DoubleVar()
        y_value = DoubleVar()
        width_value = DoubleVar()
        height_value = DoubleVar()

        ttk.Label(frame, text="Edit the selected detection target. Changes update the preview.", wraplength=290).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, padx=4, pady=(4, 10)
        )
        ttk.Label(frame, text="Target").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        selector = ttk.Combobox(frame, textvariable=roi_number, values=tuple(range(1, len(self.manual_boxes) + 1)), state="readonly", width=10)
        selector.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        for row, (label, variable) in enumerate(
            (("X", x_value), ("Y", y_value), ("Width", width_value), ("Height", height_value)),
            start=2,
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=4, pady=4)
            ttk.Entry(frame, textvariable=variable, width=16).grid(row=row, column=1, sticky="ew", padx=4, pady=4)

        def load_selected(*_args: object) -> None:
            self.selected_detection_block = roi_number.get()
            self.refresh_detection_preview()
            box = self.manual_boxes[roi_number.get() - 1]
            minimum = box.min(axis=0)
            maximum = box.max(axis=0)
            x_value.set(round(float(minimum[0]), 2))
            y_value.set(round(float(minimum[1]), 2))
            width_value.set(round(float(maximum[0] - minimum[0]), 2))
            height_value.set(round(float(maximum[1] - minimum[1]), 2))

        def apply_edit() -> None:
            x, y = x_value.get(), y_value.get()
            width, height = width_value.get(), height_value.get()
            if width < 8 or height < 8:
                messagebox.showwarning("ROI too small", "Width and height must be at least 8 pixels.", parent=dialog)
                return
            self.manual_boxes[roi_number.get() - 1] = np.array(
                [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                dtype=np.float64,
            )
            self.refresh_detection_preview()

        def delete_selected() -> None:
            index = roi_number.get() - 1
            self.manual_boxes.pop(index)
            self.excluded_blocks = {
                block_id - 1 if block_id > index + 1 else block_id
                for block_id in self.excluded_blocks
                if block_id != index + 1
            }
            dialog.destroy()
            self.refresh_detection_preview()

        selector.bind("<<ComboboxSelected>>", load_selected)
        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="Apply", command=apply_edit).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Delete ROI", command=delete_selected).pack(side=tk.LEFT, padx=(6, 0))
        def close_dialog() -> None:
            self.selected_detection_block = None
            self.refresh_detection_preview()
            dialog.destroy()

        ttk.Button(buttons, text="Close", command=close_dialog).pack(side=tk.RIGHT)
        load_selected()
        dialog.transient(self)
        dialog.grab_set()
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    def _update_raw_controls(self) -> None:
        enabled = self.raw.get()
        if hasattr(self, "reload_raw_button"):
            raw_selected = bool(self.input_files and is_raw_input_path(self.input_files[0]))
            self.reload_raw_button.configure(state=tk.NORMAL if raw_selected else tk.DISABLED)
        if hasattr(self, "raw_fields"):
            if enabled:
                self.raw_fields.pack(fill=tk.X)
            else:
                self.raw_fields.pack_forget()

        def set_enabled(widget: tk.Widget, active: bool) -> None:
            state = "readonly" if active and isinstance(widget, ttk.Combobox) else tk.NORMAL if active else tk.DISABLED
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

        for widget in self.raw_widgets:
            set_enabled(widget, enabled)
        if not enabled:
            return
        mode = normalize_raw_normalization(self.raw_normalization.get())
        for widgets, active in (
            (self.raw_bit_widgets, mode == "bit-depth"),
            (self.raw_manual_widgets, mode == "manual"),
        ):
            for widget in widgets:
                set_enabled(widget, active)

    def log(self, message: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{timestamp}  {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def current_values(self) -> dict[str, object]:
        return {
            "threshold": self.threshold.get(),
            "threshold_mode": self.threshold_mode.get(),
            "threshold_window": self.threshold_window.get(),
            "roi_radius": self.roi_radius.get(),
            "esf_method": self.esf_method.get(),
            "linear": self.linear.get(),
            "invert": self.invert.get(),
            "single_roi": self.single_roi.get(),
            "mtf_metric": self.mtf_metric.get(),
            "mtf": self.mtf.get(),
            "annotate": self.annotate.get(),
            "edges": self.edges.get(),
            "heatmap": self.heatmap.get(),
            "full_sfr": self.full_sfr.get(),
            "nosmoothing": self.nosmoothing.get(),
            "auto_tune": self.auto_tune.get(),
            "exclude_small_fiducials": self.exclude_small_fiducials.get(),
            "fiducial_max_area_percent": self.fiducial_max_area_percent.get(),
            "quality_filter": self.quality_filter.get(),
            "annotation_labels": self.annotation_labels.get(),
            "pixelsize": self.pixelsize.get(),
            "raw": self.raw.get(),
            "raw_width": self.raw_width.get(),
            "raw_height": self.raw_height.get(),
            "raw_dtype": self.raw_dtype.get(),
            "raw_byte_order": self.raw_byte_order.get(),
            "raw_header": self.raw_header.get(),
            "raw_channels": self.raw_channels.get(),
            "raw_channel_order": self.raw_channel_order.get(),
            "raw_normalization": self.raw_normalization.get(),
            "raw_bit_depth": self.raw_bit_depth.get(),
            "raw_alignment": self.raw_alignment.get(),
            "raw_black_level": self.raw_black_level.get(),
            "raw_white_level": self.raw_white_level.get(),
            "log_level": "INFO",
        }

    def apply_values(self, values: dict[str, object]) -> None:
        for name, value in values.items():
            variable = getattr(self, name, None)
            if isinstance(variable, tk.Variable):
                try:
                    variable.set(value)
                except tk.TclError:
                    pass

    def save_preset(self) -> None:
        filename = filedialog.asksaveasfilename(title="Save settings preset", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filename:
            Path(filename).write_text(json.dumps(self.current_values(), indent=2), encoding="utf-8")
            self.status.set("Settings preset saved")

    def load_preset(self) -> None:
        filename = filedialog.askopenfilename(title="Load settings preset", filetypes=[("JSON", "*.json")])
        if filename:
            self.apply_values(json.loads(Path(filename).read_text(encoding="utf-8")))
            self.status.set("Settings preset loaded")

    def save_project(self) -> None:
        filename = filedialog.asksaveasfilename(title="Save project", defaultextension=".mtfproject", filetypes=[("MTF project", "*.mtfproject")])
        if not filename:
            return
        payload = {
            "inputs": [str(path) for path in self.input_files],
            "output_root": str(self.current_output_root),
            "settings": self.current_values(),
            "manual_boxes": [box.tolist() for box in self.manual_boxes],
            "excluded_blocks": sorted(self.excluded_blocks),
            "manual_roi_active": self.manual_roi_active,
        }
        Path(filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status.set("Project saved")

    def open_project(self) -> None:
        filename = filedialog.askopenfilename(title="Open project", filetypes=[("MTF project", "*.mtfproject"), ("JSON", "*.json")])
        if not filename:
            return
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        self.input_files = [Path(path) for path in payload.get("inputs", [])]
        self.current_output_root = Path(payload.get("output_root", self.current_output_root))
        self.apply_values(payload.get("settings", {}))
        self.manual_boxes = [np.asarray(box, dtype=np.float64) for box in payload.get("manual_boxes", [])]
        self.excluded_blocks = set(payload.get("excluded_blocks", []))
        self.manual_roi_active = bool(payload.get("manual_roi_active", bool(self.manual_boxes)))
        self.input_label.config(text=f"Input: {len(self.input_files)} file(s)" if self.input_files else "Input: none")
        self.output_label.config(text=f"Output: {short_path(self.current_output_root)}")
        if self.input_files:
            self.reload_original_preview(show_errors=False)
        self.update_workflow_actions()
        self.status.set("Project loaded")

    def run_analysis(self) -> None:
        if not self.input_files:
            messagebox.showwarning("No input", "Select at least one input image first.")
            return
        if not self.annotate.get() and not self.edges.get() and not self.heatmap.get():
            messagebox.showwarning("No outputs", "Select at least one output.")
            return
        if self.roi_radius.get() < 4.0:
            messagebox.showwarning("Invalid ROI size", "Edge ROI radius must be at least 4 pixels.")
            return
        if self.exclude_small_fiducials.get() and not 0.0 < self.fiducial_max_area_percent.get() <= 100.0:
            messagebox.showwarning("Invalid fiducial filter", "Minimum fiducial area must be between 0 and 100 percent.")
            return
        self.run_button.config(state=tk.DISABLED)
        self.progress.start(12)
        self.status.set(f"Analyzing {len(self.input_files)} image(s)...")
        self.summary_title.set("Analysis in progress")
        self.summary_detail.set("Detecting targets and calculating edge response.")
        self.log(f"Starting analysis for {len(self.input_files)} file(s)")
        values = self.current_values()
        manual_active = len(self.input_files) == 1 and self.manual_roi_active
        values["manual_boxes"] = [box.tolist() for box in self.manual_boxes] if manual_active else None
        values["excluded_blocks"] = sorted(self.excluded_blocks) if manual_active else []
        thread = threading.Thread(target=self._run_worker, args=(self.input_files.copy(), values), daemon=True)
        thread.start()

    def _run_worker(self, files: list[Path], values: dict[str, object]) -> None:
        for input_path in files:
            original_preview_path: Path | None = None
            try:
                output_dir = self.current_output_root / input_path.stem
                preview_dir = Path(tempfile.gettempdir()) / "mtf_mapper_python_gui_previews" / input_path.stem
                original_preview_path = prepare_original_preview(input_path, preview_dir, values)
                self.worker_queue.put(("original", (input_path, original_preview_path)))
                args = namespace_from_gui_values(input_path, output_dir, values)
                lum_for_detection, original_for_detection = mtf_mapper_py.load_input_luminance(args)
                if args.manual_boxes is not None:
                    excluded = set(args.excluded_blocks)
                    report = mtf_mapper_py.DetectionReport(
                        "manual",
                        args.threshold,
                        args.threshold_window,
                        accepted_count=sum(
                            block_id not in excluded for block_id in range(1, len(args.manual_boxes) + 1)
                        ),
                    )
                elif args.auto_tune:
                    _boxes, report = mtf_mapper_py.auto_tune_detection(
                        lum_for_detection,
                        args.fiducial_max_area_ratio if args.exclude_small_fiducials else 0.0,
                    )
                else:
                    _boxes, report = mtf_mapper_py.detect_boxes_with_diagnostics(
                        lum_for_detection,
                        args.threshold,
                        args.threshold_window,
                        args.threshold_mode,
                        args.fiducial_max_area_ratio if args.exclude_small_fiducials else 0.0,
                    )
                lum, annotated, measurements = mtf_mapper_py.analyze_image(args)
                quality_filter = str(values.get("quality_filter", "All edges"))
                if quality_filter == "Good only":
                    measurements = [m for m in measurements if m.quality_label == "Good"]
                elif quality_filter == "Good + Review":
                    measurements = [m for m in measurements if m.quality_label != "Poor"]
                if not measurements:
                    raise ValueError("All measured edges were removed by the quality filter")
                annotated = mtf_mapper_py.make_annotation(
                    lum, original_for_detection, measurements, str(values.get("annotation_labels", "All values"))
                )
                mtf_mapper_py.prepare_output_dir(output_dir, args.annotate, args.edges, args.heatmap)
                if args.edges:
                    mtf_mapper_py.write_edge_tables(output_dir, measurements)
                if args.annotate:
                    mtf_mapper_py.write_annotation(output_dir, annotated)
                    annotated_preview_path = output_dir / "annotated.png"
                else:
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    annotated_preview_path = preview_dir / "annotated_preview.png"
                    if not mtf_mapper_py.cv2.imwrite(str(annotated_preview_path), annotated):
                        raise ValueError(f"Cannot create annotated preview for {input_path}")
                if args.heatmap:
                    mtf_mapper_py.write_heatmap(output_dir, lum, measurements)
                    heatmap_preview_path = output_dir / "mtf_heatmap.png"
                else:
                    heatmap_preview_path = None
                mtf_mapper_py.write_diagnostics(
                    output_dir, report, measurements, getattr(args, "raw_normalization_report", None)
                )
                self.worker_queue.put(
                    (
                        "result",
                        GuiRunResult(
                            input_path,
                            output_dir,
                            len(measurements),
                            measurements,
                            original_preview_path,
                            annotated_preview_path,
                            heatmap_preview_path,
                            report,
                        ),
                    )
                )
            except Exception as exc:
                self.worker_queue.put(("error", GuiRunError(input_path, original_preview_path, exc)))
        self.worker_queue.put(("done", None))

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "original":
                    _input_path, original_preview_path = payload  # type: ignore[misc]
                    self.set_preview_sources(original_preview_path, None, [])
                elif kind == "result":
                    self.add_result(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    error = payload  # type: ignore[assignment]
                    raw_input_failed = is_raw_input_path(error.input_path)
                    self.status.set("Check Raw import settings" if raw_input_failed else "Analysis failed")
                    self.summary_title.set(f"{error.input_path.name} - analysis failed")
                    if raw_input_failed:
                        self.summary_detail.set(
                            "Check Raw import metadata in Advanced, reload the image, then run analysis again."
                        )
                    else:
                        self.summary_detail.set("The original image is still available. Adjust the settings and run again.")
                    self.summary_edges.set("-")
                    self.summary_blocks.set("-")
                    self.summary_median.set("-")
                    self.summary_range.set("-")
                    self.selected_measurement = None
                    self.curve_plot_state = None
                    self.selected_edge.set("No edge selected")
                    self.close_edge_inspector()
                    self.log(f"Analysis failed for {error.input_path.name}: {error.error}")
                    if error.original_preview_path is not None:
                        self.set_preview_sources(error.original_preview_path, None, [])
                    if raw_input_failed:
                        self.show_dock_tab(self.advanced_tab)
                        messagebox.showwarning(
                            "Check Raw import settings",
                            raw_import_error_message(error.input_path, error.error),
                        )
                    else:
                        messagebox.showerror("Analysis failed", str(error.error))
                elif kind == "done":
                    self.run_button.config(state=tk.NORMAL)
                    self.progress.stop()
                    self.status.set("Analysis complete")
                    self.log("Analysis complete")
                    self.update_workflow_actions()
        except queue.Empty:
            pass
        self.after(100, self._poll_worker_queue)

    def add_result(self, result: GuiRunResult) -> None:
        parent_id = self.result_tree.insert("", tk.END, text=f"{result.input_path.name} ({result.edge_count} edges)")
        self.result_measurements[result.output_dir] = result.measurements
        self.detection_report = result.detection_report
        for path in sorted(result.output_dir.glob("*")):
            item_id = self.result_tree.insert(parent_id, tk.END, text=path.name)
            self.result_rows[item_id] = path
        self.result_tree.item(parent_id, open=True)
        self.log(f"{result.input_path.name}: measured {result.edge_count} edges")
        summary = summarize_measurements(result.measurements)
        metric = result.measurements[0].mtf_column if result.measurements else self.mtf_metric.get()
        self.summary_title.set(f"{result.input_path.name} - analysis complete")
        if result.measurements:
            self.summary_detail.set(f"Measured {metric}. Click an edge to inspect SFR, ESF, and LSF.")
            self.summary_edges.set(str(summary["edges"]))
            self.summary_blocks.set(str(summary["blocks"]))
            self.summary_median.set(f"{summary['median']:.3f}")
            self.summary_range.set(f"{summary['minimum']:.3f}-{summary['maximum']:.3f}")
        else:
            self.summary_detail.set("No valid slanted edges were detected. Check the image and detection settings.")
            self.summary_edges.set("0")
            self.summary_blocks.set("0")
            self.summary_median.set("-")
            self.summary_range.set("-")
        self.show_dock_tab(self.results_tab)
        self.set_diagnostics(result.detection_report, result.measurements, source="Last completed analysis")
        self.set_preview_sources(
            result.original_preview_path,
            result.annotated_preview_path,
            result.measurements,
            "Annotated",
            result.heatmap_preview_path,
        )
        self.after(100, self.fit_preview)
        self.update_workflow_actions()
        self.write_batch_summary()

    def set_diagnostics(
        self,
        report: mtf_mapper_py.DetectionReport,
        measurements: list[mtf_mapper_py.EdgeMeasurement],
        source: str = "Diagnostics",
    ) -> None:
        self.diagnostics_source.set(source)
        counts = {label: sum(m.quality_label == label for m in measurements) for label in ("Good", "Review", "Poor")}
        lines = [
            "DETECTION",
            f"Mode: {report.threshold_mode}",
            f"Threshold: {report.threshold:.3f}   Adaptive window: {report.threshold_window:.3f}",
            f"Contours: {report.contour_count}   Accepted targets: {report.accepted_count}",
            f"Rejected: {report.rejected_small_area} small, {report.rejected_short_side} short, "
            f"{report.rejected_shape} non-rectangular, {report.rejected_fiducial} fiducial-sized",
            "",
            "EDGE QUALITY",
            f"Edge quality: {counts['Good']} good, {counts['Review']} review, {counts['Poor']} poor",
            f"ESF methods: {sum(m.esf_method == 'pixel-binned' for m in measurements)} pixel-binned, "
            f"{sum(m.esf_method == 'interpolated' for m in measurements)} interpolated",
            "",
            "RECOMMENDATION",
            *report.suggestions(),
        ]
        self.diagnostics_text.config(state=tk.NORMAL)
        self.diagnostics_text.delete("1.0", tk.END)
        self.diagnostics_text.insert(tk.END, "\n".join(lines))
        self.diagnostics_text.config(state=tk.DISABLED)

    def write_batch_summary(self) -> None:
        if len(self.result_measurements) < 2:
            return
        self.current_output_root.mkdir(parents=True, exist_ok=True)
        with (self.current_output_root / "batch_summary.csv").open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["output", "edges", "blocks", "median", "minimum", "maximum", "good_edges"])
            for output_dir, measurements in self.result_measurements.items():
                summary = summarize_measurements(measurements)
                writer.writerow([
                    output_dir.name,
                    summary["edges"],
                    summary["blocks"],
                    summary["median"],
                    summary["minimum"],
                    summary["maximum"],
                    sum(m.quality_label == "Good" for m in measurements),
                ])

    def clear_results(self) -> None:
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_rows.clear()
        self.result_measurements.clear()
        self.clear_preview("Open an image to run analysis")
        self.preview_image = None
        self.preview_rgb = None
        self.preview_rgb_path = None
        self.preview_state = None
        self.preview_path = None
        self.preview_measurements = []
        self.original_preview_path = None
        self.annotated_preview_path = None
        self.annotated_preview_measurements = []
        self.heatmap_preview_path = None
        self.detection_preview_path = None
        self.threshold_preview_path = None
        self.manual_boxes = []
        self.excluded_blocks.clear()
        self.manual_roi_active = False
        self.preview_mode.set("Original")
        self.preview_mode_box.configure(values=("Original",))
        self.selected_measurement = None
        self.curve_plot_state = None
        self.selected_edge.set("No edge selected")
        self.close_edge_inspector()
        self.status.set("Results cleared")
        self.summary_title.set("No analysis yet")
        self.summary_detail.set("Open an image or try the sample chart to see a measurement summary.")
        self.summary_edges.set("-")
        self.summary_blocks.set("-")
        self.summary_median.set("-")
        self.summary_range.set("-")
        self.diagnostics_source.set("No diagnostics yet")
        self.preview_guide.set("")
        self.log("Results cleared")
        self.update_workflow_actions()

    def on_result_selected(self, _event: tk.Event) -> None:
        selected = self.result_tree.selection()
        if not selected:
            return
        path = self.result_rows.get(selected[0])
        if path is None:
            return
        if path.suffix.lower() == ".png":
            self.show_image(path, self.result_measurements.get(path.parent, []))
        elif path.suffix.lower() == ".csv":
            self.status.set(f"Selected {path.name}")

    def open_selected_result(self) -> None:
        selected = self.result_tree.selection()
        if not selected:
            return
        path = self.result_rows.get(selected[0])
        if path is not None:
            open_with_system(path)

    def open_output_folder(self) -> None:
        open_with_system(self.current_output_root)

    def clear_preview(self, text: str) -> None:
        self.preview.delete("all")
        self.preview_info.set("No image")
        self.preview_x_scroll.grid_remove()
        self.preview_y_scroll.grid_remove()
        self.preview.configure(scrollregion=(0, 0, max(self.preview.winfo_width(), 320), max(self.preview.winfo_height(), 200)))
        self.preview.create_text(
            max(self.preview.winfo_width() // 2, 160),
            max(self.preview.winfo_height() // 2, 100),
            text=text,
            fill="#555555",
            tags="placeholder",
        )

    def calculate_fit_scale(self, path: Path) -> float:
        if self.preview_rgb_path == path and self.preview_rgb is not None:
            image_height, image_width = self.preview_rgb.shape[:2]
        else:
            image_width, image_height = image_size_from_cv(path)
        canvas_width = max(self.preview.winfo_width() - 18, 320)
        canvas_height = max(self.preview.winfo_height() - 18, 220)
        return max(0.05, min(canvas_width / image_width, canvas_height / image_height, 1.0))

    def set_preview_sources(
        self,
        original_path: Path,
        annotated_path: Path | None,
        measurements: list[mtf_mapper_py.EdgeMeasurement],
        preferred_mode: str = "Original",
        heatmap_path: Path | None = None,
    ) -> None:
        self.preview_rgb = None
        self.preview_rgb_path = None
        self.original_preview_path = original_path
        self.annotated_preview_path = annotated_path
        self.annotated_preview_measurements = measurements
        self.heatmap_preview_path = heatmap_path
        modes = ["Original"]
        if annotated_path is not None:
            modes.append("Annotated")
        if heatmap_path is not None:
            modes.append("Spatial map")
        if self.detection_preview_path is not None:
            modes.append("Detection")
        if self.threshold_preview_path is not None:
            modes.append("Threshold mask")
        modes = tuple(modes)
        self.preview_mode_box.configure(values=modes)
        mode = preferred_mode if preferred_mode in modes else "Original"
        if self.preview_mode.get() == mode:
            self.show_preview_mode()
        else:
            self.preview_mode.set(mode)

    def show_preview_mode(self) -> None:
        mode = self.preview_mode.get()
        if mode == "Detection":
            self.preview_guide.set(
                "Cyan = included target. Gray = excluded. Click a target to toggle it; Shift-drag to add an ROI."
            )
        elif mode == "Threshold mask":
            self.preview_guide.set("White regions pass the current dark-object threshold; black regions do not.")
        elif mode == "Annotated":
            self.preview_guide.set("Click near an annotated edge to inspect its SFR, ESF, and LSF curves.")
        else:
            self.preview_guide.set("")
        if mode == "Annotated" and self.annotated_preview_path is not None:
            self.show_image(self.annotated_preview_path, self.annotated_preview_measurements)
        elif mode == "Spatial map" and self.heatmap_preview_path is not None:
            self.show_image(self.heatmap_preview_path, [])
        elif mode == "Detection" and self.detection_preview_path is not None:
            self.show_image(self.detection_preview_path, [])
        elif mode == "Threshold mask" and self.threshold_preview_path is not None:
            self.show_image(self.threshold_preview_path, [])
        elif self.original_preview_path is not None:
            self.show_image(self.original_preview_path, [])

    def show_image(self, path: Path, measurements: list[mtf_mapper_py.EdgeMeasurement] | None = None) -> None:
        self.cancel_pending_preview_zoom()
        self.preview_path = path
        self.preview_measurements = measurements or []
        try:
            if self.preview_rgb_path != path or self.preview_rgb is None:
                self.preview_rgb = rgb_image_from_cv(path)
                self.preview_rgb_path = path
            self.preview_fit_scale = self.calculate_fit_scale(path)
        except ValueError:
            self.preview_rgb = None
            self.preview_rgb_path = None
            self.preview_fit_scale = 1.0
        self.preview_zoom = self.preview_fit_scale
        self.render_preview(reset_view=True)

    def render_preview(self, reset_view: bool = False) -> None:
        if self.preview_path is None:
            self.clear_preview("Open an image to run analysis")
            return
        try:
            if self.preview_rgb_path != self.preview_path or self.preview_rgb is None:
                self.preview_rgb = rgb_image_from_cv(self.preview_path)
                self.preview_rgb_path = self.preview_path
            image, image_width, image_height, display_width, display_height = photo_image_from_rgb(
                self.preview_rgb, self.preview_zoom
            )
            self.preview_image = image
            canvas_width = max(self.preview.winfo_width(), 1)
            canvas_height = max(self.preview.winfo_height(), 1)
            scroll_width = max(display_width, canvas_width)
            scroll_height = max(display_height, canvas_height)
            offset_x = max((canvas_width - display_width) // 2, 0)
            offset_y = max((canvas_height - display_height) // 2, 0)
            self.preview_state = PreviewState(
                self.preview_path,
                self.preview_measurements,
                self.preview_zoom,
                image_width,
                image_height,
                offset_x,
                offset_y,
            )
            self.preview.delete("all")
            self.preview.create_rectangle(
                offset_x - 2,
                offset_y - 2,
                offset_x + display_width + 2,
                offset_y + display_height + 2,
                fill="#ffffff",
                outline="#5f6875",
                width=2,
            )
            self.preview.create_image(offset_x, offset_y, image=image, anchor=tk.NW)
            self.preview.create_rectangle(
                offset_x,
                offset_y,
                offset_x + display_width,
                offset_y + display_height,
                outline="#3f4854",
                width=1,
            )
            self.preview.configure(scrollregion=(0, 0, scroll_width, scroll_height))
            if display_width <= canvas_width:
                self.preview_x_scroll.grid_remove()
            else:
                self.preview_x_scroll.grid()
            if display_height <= canvas_height:
                self.preview_y_scroll.grid_remove()
            else:
                self.preview_y_scroll.grid()
            self.draw_selected_edge_highlight()
            if reset_view:
                self.preview.xview_moveto(0.0)
                self.preview.yview_moveto(0.0)
            hint = " - click near an edge to view SFR" if self.preview_measurements else ""
            self.status.set(f"Previewing {self.preview_path.name}{hint}")
            self.preview_info.set(
                f"{image_width} x {image_height}  |  {self.preview_zoom * 100:.0f}%"
            )
        except (tk.TclError, ValueError) as exc:
            self.clear_preview(f"Cannot preview {self.preview_path.name}")
            self.log(str(exc))
            self.preview_image = None
            self.preview_rgb = None
            self.preview_rgb_path = None
            self.preview_state = None

    def fit_preview(self) -> None:
        if self.preview_path is None:
            return
        self.cancel_pending_preview_zoom()
        self.preview_fit_scale = self.calculate_fit_scale(self.preview_path)
        self.preview_zoom = self.preview_fit_scale
        self.render_preview(reset_view=True)

    def zoom_preview(self, factor: float, anchor_x: int | None = None, anchor_y: int | None = None) -> None:
        if self.preview_path is None:
            return
        state = self.preview_state
        if anchor_x is None:
            anchor_x = self.preview.winfo_width() // 2
        if anchor_y is None:
            anchor_y = self.preview.winfo_height() // 2
        image_anchor: tuple[float, float] | None = None
        if state is not None:
            canvas_x = int(self.preview.canvasx(anchor_x))
            canvas_y = int(self.preview.canvasy(anchor_y))
            image_anchor = preview_to_image_coords(canvas_x, canvas_y, state)
        self.preview_zoom = min(max(self.preview_zoom * factor, 0.05), 8.0)
        self.render_preview(reset_view=False)
        new_state = self.preview_state
        if image_anchor is None or new_state is None:
            return
        target_x = new_state.offset_x + image_anchor[0] * new_state.display_scale
        target_y = new_state.offset_y + image_anchor[1] * new_state.display_scale
        region = self.preview.cget("scrollregion").split()
        if len(region) == 4:
            region_width = max(float(region[2]) - float(region[0]), 1.0)
            region_height = max(float(region[3]) - float(region[1]), 1.0)
            self.preview.xview_moveto(max(0.0, (target_x - anchor_x) / region_width))
            self.preview.yview_moveto(max(0.0, (target_y - anchor_y) / region_height))

    def schedule_preview_zoom(self, factor: float, anchor_x: int, anchor_y: int) -> None:
        self.pending_zoom_factor *= factor
        self.pending_zoom_anchor = (anchor_x, anchor_y)
        if self.pending_zoom_after is not None:
            self.after_cancel(self.pending_zoom_after)
        self.pending_zoom_after = self.after(35, self.apply_pending_preview_zoom)

    def cancel_pending_preview_zoom(self) -> None:
        if self.pending_zoom_after is not None:
            self.after_cancel(self.pending_zoom_after)
        self.pending_zoom_factor = 1.0
        self.pending_zoom_anchor = None
        self.pending_zoom_after = None

    def schedule_center_zoom(self, factor: float) -> None:
        self.schedule_preview_zoom(factor, self.preview.winfo_width() // 2, self.preview.winfo_height() // 2)

    def apply_pending_preview_zoom(self) -> None:
        factor = self.pending_zoom_factor
        anchor = self.pending_zoom_anchor
        self.pending_zoom_factor = 1.0
        self.pending_zoom_anchor = None
        self.pending_zoom_after = None
        if anchor is not None:
            self.zoom_preview(factor, anchor[0], anchor[1])

    def on_preview_wheel(self, event: tk.Event) -> None:
        if self.preview_path is None:
            return
        self.schedule_preview_zoom(wheel_zoom_factor(float(event.delta)), event.x, event.y)

    def on_preview_magnify(self, event: tk.Event) -> None:
        if self.preview_path is None:
            return
        self.schedule_preview_zoom(magnify_zoom_factor(float(event.delta)), event.x, event.y)

    def on_preview_configure(self, _event: tk.Event) -> None:
        if self.pending_render_after is not None:
            self.after_cancel(self.pending_render_after)
        self.pending_render_after = self.after(60, self.apply_pending_preview_render)

    def apply_pending_preview_render(self) -> None:
        self.pending_render_after = None
        if self.preview_path is None:
            self.clear_preview("Open an image to run analysis")
        else:
            self.render_preview(reset_view=False)

    def on_preview_press(self, event: tk.Event) -> None:
        self.preview_drag_start = (event.x, event.y)
        self.preview.scan_mark(event.x, event.y)

    def on_preview_drag(self, event: tk.Event) -> None:
        self.preview.scan_dragto(event.x, event.y, gain=1)

    def on_preview_release(self, event: tk.Event) -> None:
        if self.preview_drag_start is None:
            return
        start_x, start_y = self.preview_drag_start
        self.preview_drag_start = None
        drag_distance = math.hypot(event.x - start_x, event.y - start_y)
        if drag_distance > 6 and (event.state & 0x0001) and self.preview_mode.get() == "Detection":
            self.add_manual_roi(start_x, start_y, event.x, event.y)
            return
        if drag_distance > 6:
            return
        self.on_preview_click(event)

    def add_manual_roi(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        state = self.preview_state
        if state is None:
            return
        x0, y0 = preview_to_image_coords(int(self.preview.canvasx(start_x)), int(self.preview.canvasy(start_y)), state)
        x1, y1 = preview_to_image_coords(int(self.preview.canvasx(end_x)), int(self.preview.canvasy(end_y)), state)
        left, right = sorted((max(0.0, x0), min(float(state.image_width), x1)))
        top, bottom = sorted((max(0.0, y0), min(float(state.image_height), y1)))
        if right - left < 8 or bottom - top < 8:
            return
        self.manual_boxes.append(np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float64))
        self.manual_roi_active = True
        self.refresh_detection_preview()

    def refresh_detection_preview(self) -> None:
        if not self.input_files:
            return
        args = namespace_from_gui_values(self.input_files[0], self.current_output_root / self.input_files[0].stem, self.current_values())
        lum, original = mtf_mapper_py.load_input_luminance(args)
        preview = mtf_mapper_py.make_detection_preview(
            lum,
            original,
            self.manual_boxes,
            sorted(self.excluded_blocks),
            self.selected_detection_block,
        )
        if self.detection_preview_path is None:
            return
        mtf_mapper_py.cv2.imwrite(str(self.detection_preview_path), preview)
        self.preview_rgb = None
        self.preview_rgb_path = None
        self.show_image(self.detection_preview_path, [])

    def on_preview_click(self, event: tk.Event) -> None:
        state = self.preview_state
        if state is None:
            return
        canvas_x = int(self.preview.canvasx(event.x))
        canvas_y = int(self.preview.canvasy(event.y))
        image_x, image_y = preview_to_image_coords(canvas_x, canvas_y, state)
        if image_x < 0 or image_y < 0:
            return
        if image_x > state.image_width or image_y > state.image_height:
            return
        if self.preview_mode.get() == "Detection" and self.manual_boxes:
            block_id = min(
                range(1, len(self.manual_boxes) + 1),
                key=lambda idx: float(np.linalg.norm(self.manual_boxes[idx - 1].mean(axis=0) - np.array([image_x, image_y]))),
            )
            if block_id in self.excluded_blocks:
                self.excluded_blocks.remove(block_id)
            else:
                self.excluded_blocks.add(block_id)
            self.refresh_detection_preview()
            self.update_workflow_actions()
            return
        if not state.measurements:
            return
        measurement = nearest_measurement(state.measurements, image_x, image_y)
        if measurement is None:
            return
        distance = distance_to_measurement(measurement, image_x, image_y)
        if distance > 72.0:
            self.status.set("Click closer to an annotated edge to inspect its curves")
            return
        self.show_edge_curves(measurement)

    def show_edge_curves(self, measurement: mtf_mapper_py.EdgeMeasurement) -> None:
        self.selected_measurement = measurement
        self.selected_edge.set(
            f"Block {measurement.block_id} | Edge ({measurement.edge_x:.1f}, {measurement.edge_y:.1f})\n"
            f"{measurement.mtf_column}={measurement.mtf_value:.4g} | {measurement.quality_label} "
            f"{measurement.quality_score:.0%} | {measurement.esf_method} ESF"
        )
        self.draw_selected_edge_highlight()
        self.open_edge_inspector()
        self.redraw_selected_curve()
        self.status.set(f"Showing {self.curve_type.get()} curve for selected edge")

    def open_edge_inspector(self) -> None:
        if self.edge_inspector is not None and self.edge_inspector.winfo_exists():
            self.edge_inspector.deiconify()
            self.edge_inspector.lift()
            return

        inspector = tk.Toplevel(self)
        inspector.title("Edge Inspector")
        inspector.geometry("760x620")
        inspector.minsize(560, 420)
        inspector.protocol("WM_DELETE_WINDOW", self.close_edge_inspector)
        inspector.bind("<Escape>", lambda _event: self.close_edge_inspector())
        self.edge_inspector = inspector

        header = ttk.Frame(inspector, padding=(12, 10))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            textvariable=self.selected_edge,
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(header, text="Curve").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Combobox(
            header,
            textvariable=self.curve_type,
            values=("SFR", "ESF", "LSF"),
            state="readonly",
            width=6,
        ).pack(side=tk.LEFT)
        ttk.Button(header, text="Export CSV...", command=self.export_selected_curve).pack(side=tk.LEFT, padx=(8, 0))

        self.sfr_canvas = tk.Canvas(
            inspector,
            background="white",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
        )
        self.sfr_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.sfr_canvas.bind("<Configure>", lambda _event: self.redraw_selected_curve())
        self.sfr_canvas.bind("<Motion>", self.on_sfr_hover)
        self.sfr_canvas.bind("<Leave>", self.on_sfr_leave)

    def export_selected_curve(self) -> None:
        measurement = self.selected_measurement
        if measurement is None:
            return
        curve_type = self.curve_type.get()
        values, x_min, x_max, _x_label, _y_label, x_unit = curve_data(measurement, curve_type)
        filename = filedialog.asksaveasfilename(
            title=f"Export {curve_type} curve",
            defaultextension=".csv",
            initialfile=f"block_{measurement.block_id}_{curve_type.lower()}.csv",
            filetypes=[("CSV", "*.csv")],
            parent=self.edge_inspector,
        )
        if not filename:
            return
        x_values = np.linspace(x_min, x_max, len(values))
        with Path(filename).open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow([f"x ({x_unit})", curve_type])
            writer.writerows(zip(x_values, values))
        self.status.set(f"Exported {curve_type} curve")

    def close_edge_inspector(self) -> None:
        if self.edge_inspector is not None and self.edge_inspector.winfo_exists():
            self.edge_inspector.destroy()
        self.edge_inspector = None
        self.sfr_canvas = None
        self.curve_plot_state = None

    def draw_selected_edge_highlight(self) -> None:
        self.preview.delete("selected-edge")
        state = self.preview_state
        measurement = self.selected_measurement
        if state is None or measurement is None or measurement not in state.measurements:
            return
        x0 = state.offset_x + measurement.edge_start_x * state.display_scale
        y0 = state.offset_y + measurement.edge_start_y * state.display_scale
        x1 = state.offset_x + measurement.edge_end_x * state.display_scale
        y1 = state.offset_y + measurement.edge_end_y * state.display_scale
        self.preview.create_line(x0, y0, x1, y1, fill="#00ffff", width=5, tags="selected-edge")

    def redraw_selected_curve(self) -> None:
        if self.sfr_canvas is None or not self.sfr_canvas.winfo_exists():
            self.curve_plot_state = None
            return
        self.sfr_canvas.delete("all")
        if self.selected_measurement is None:
            self.curve_plot_state = None
            self.sfr_canvas.create_text(
                max(self.sfr_canvas.winfo_width() // 2, 160),
                max(self.sfr_canvas.winfo_height() // 2, 100),
                text="Click an annotated edge to inspect its SFR, ESF, and LSF curves",
                fill="#555555",
            )
            return
        self.curve_plot_state = draw_curve(self.sfr_canvas, self.selected_measurement, self.curve_type.get())
        self.status.set(f"Showing {self.curve_type.get()} curve for selected edge")

    def on_sfr_hover(self, event: tk.Event) -> None:
        state = self.curve_plot_state
        if state is None or self.sfr_canvas is None:
            return
        self.sfr_canvas.delete("hover")
        if event.x < state.x0 or event.x > state.x1 or event.y < state.y1 or event.y > state.y0:
            return
        frac = (event.x - state.x0) / max(state.x1 - state.x0, 1.0)
        x_value = state.x_min + frac * (state.x_max - state.x_min)
        sample_index = frac * (len(state.values) - 1)
        value = float(np.interp(sample_index, np.arange(len(state.values)), state.values))
        y = state.y0 - (value - state.y_min) / (state.y_max - state.y_min) * (state.y0 - state.y1)
        self.sfr_canvas.create_line(event.x, state.y1, event.x, state.y0, fill="#d12b2b", dash=(5, 4), tags="hover")
        self.sfr_canvas.create_oval(event.x - 3, y - 3, event.x + 3, y + 3, fill="#d12b2b", outline="", tags="hover")
        label = f"x={x_value:.4f} {state.x_unit}, y={value:.4f}"
        label_x = min(max(event.x + 10, state.x0 + 4), state.x1 - 225)
        label_y = state.y1 + 10
        self.sfr_canvas.create_rectangle(label_x - 4, label_y - 4, label_x + 222, label_y + 18, fill="white", outline="#b0b0b0", tags="hover")
        self.sfr_canvas.create_text(label_x, label_y, text=label, anchor=tk.NW, fill="#333333", tags="hover")

    def on_sfr_leave(self, _event: tk.Event) -> None:
        if self.sfr_canvas is not None:
            self.sfr_canvas.delete("hover")

    def show_about(self) -> None:
        messagebox.showinfo(
            "About MTF Mapper Python",
            "MTF Mapper Python\n\nCore slanted-edge analyzer with annotated image and CSV outputs.",
        )


def main() -> int:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print("usage: mtf-mapper-gui\n\nLaunch the MTF Mapper Python graphical interface.")
        return 0
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = MtfMapperGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
