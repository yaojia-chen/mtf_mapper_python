#!/usr/bin/env python3
"""Tkinter GUI for the standalone Python MTF Mapper port."""

from __future__ import annotations

import argparse
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
PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_CHART = PROJECT_DIR / "samples" / "mtf_test_chart.png"


@dataclass
class GuiRunResult:
    input_path: Path
    output_dir: Path
    edge_count: int
    measurements: list[mtf_mapper_py.EdgeMeasurement]


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
class SfrPlotState:
    measurement: mtf_mapper_py.EdgeMeasurement
    x0: float
    y0: float
    x1: float
    y1: float
    y_max: float
    max_frequency: float


def namespace_from_gui_values(input_path: Path, output_dir: Path, values: dict[str, object]) -> argparse.Namespace:
    raw_enabled = bool(values.get("raw", False))
    args = argparse.Namespace(
        input_image=str(input_path),
        output_dir=str(output_dir),
        threshold=float(values.get("threshold", 0.55)),
        threshold_window=float(values.get("threshold_window", 1.0 / 3.0)),
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
        log_level=str(values.get("log_level", "INFO")),
    )
    if args.pixelsize in ("", None):
        args.pixelsize = None
    else:
        args.pixelsize = float(args.pixelsize)
    if raw_enabled:
        args.raw_width = int(args.raw_width)
        args.raw_height = int(args.raw_height)
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


def photo_image_from_cv(path: Path, display_scale: float) -> tuple[tk.PhotoImage, int, int, int, int]:
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
    height, width = rgb.shape[:2]
    display_width = max(1, int(round(width * display_scale)))
    display_height = max(1, int(round(height * display_scale)))
    interpolation = cv2.INTER_AREA if display_scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (display_width, display_height), interpolation=interpolation)
    ok, data = cv2.imencode(".ppm", resized)
    if not ok:
        raise ValueError(f"Cannot render preview for {path}")
    return tk.PhotoImage(data=data.tobytes(), format="PPM"), width, height, display_width, display_height


def draw_sfr_curve(canvas: tk.Canvas, measurement: mtf_mapper_py.EdgeMeasurement) -> SfrPlotState | None:
    canvas.delete("all")
    width = max(canvas.winfo_width(), 320)
    height = max(canvas.winfo_height(), 220)
    left, right, top, bottom = 58, 20, 20, 46
    plot_w = max(width - left - right, 1)
    plot_h = max(height - top - bottom, 1)
    x0, y0 = left, height - bottom
    x1, y1 = width - right, top

    canvas.create_rectangle(x0, y1, x1, y0, outline="#a0a0a0")
    for tick in range(5):
        frac = tick / 4
        y = y0 - frac * plot_h
        canvas.create_line(x0, y, x1, y, fill="#ececec")
        canvas.create_text(x0 - 8, y, text=f"{frac:.2f}", anchor=tk.E, fill="#555555", font=("TkDefaultFont", 9))
    for tick in range(5):
        frac = tick / 4
        x = x0 + frac * plot_w
        freq = frac if len(measurement.sfr) <= 64 else frac * 2.0
        canvas.create_line(x, y0, x, y0 + 4, fill="#666666")
        canvas.create_text(x, y0 + 18, text=f"{freq:.2f}", anchor=tk.N, fill="#555555", font=("TkDefaultFont", 9))

    sfr = measurement.sfr
    if len(sfr) < 2:
        return None
    sfr_max = max(float(v) for v in sfr)
    y_max = max(sfr_max, 1.0) if sfr_max > 0 else 1.0
    points: list[float] = []
    for idx, value in enumerate(sfr):
        x = x0 + (idx / (len(sfr) - 1)) * plot_w
        y = y0 - min(max(float(value), 0.0), y_max) / y_max * plot_h
        points.extend([x, y])
    canvas.create_line(*points, fill="#0b63ce", width=2, smooth=True)
    canvas.create_text(width // 2, height - 8, text="Frequency (cycles/pixel)", anchor=tk.S, fill="#333333")
    canvas.create_text(12, top, text="SFR", anchor=tk.NW, fill="#333333")
    return SfrPlotState(measurement, x0, y0, x1, y1, y_max, (len(sfr) - 1) / 64.0)


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
        self.preview_state: PreviewState | None = None
        self.preview_path: Path | None = None
        self.preview_measurements: list[mtf_mapper_py.EdgeMeasurement] = []
        self.preview_zoom = 1.0
        self.preview_fit_scale = 1.0
        self.preview_drag_start: tuple[int, int] | None = None
        self.selected_measurement: mtf_mapper_py.EdgeMeasurement | None = None
        self.sfr_plot_state: SfrPlotState | None = None
        self.raw_widgets: list[tk.Widget] = []
        self.current_output_root = Path(tempfile.gettempdir()) / "mtf_mapper_python_gui"

        self._build_vars()
        self._build_menu()
        self._build_layout()
        self._update_raw_controls()
        self.after(100, self._poll_worker_queue)

    def _build_vars(self) -> None:
        self.threshold = DoubleVar(value=0.55)
        self.threshold_window = DoubleVar(value=1.0 / 3.0)
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
        self.pixelsize = StringVar(value="")
        self.raw = BooleanVar(value=False)
        self.raw_width = StringVar(value="")
        self.raw_height = StringVar(value="")
        self.raw_dtype = StringVar(value="uint16")
        self.raw_byte_order = StringVar(value="little")
        self.raw_header = IntVar(value=0)
        self.raw_channels = IntVar(value=1)
        self.status = StringVar(value="Ready")
        self.selected_edge = StringVar(value="No edge selected")
        self.raw.trace_add("write", lambda *_args: self._update_raw_controls())

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open...", accelerator="Ctrl+O", command=self.open_files)
        file_menu.add_command(label="Open single edge image...", accelerator="Ctrl+E", command=self.open_single_roi)
        file_menu.add_separator()
        file_menu.add_command(label="Choose output directory...", command=self.choose_output_dir)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", accelerator="Ctrl+Q", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)
        self.bind("<Control-o>", lambda _event: self.open_files())
        self.bind("<Control-e>", lambda _event: self.open_single_roi())
        self.bind("<Control-q>", lambda _event: self.destroy())

    def _build_layout(self) -> None:
        self._build_toolbar()

        root = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        sidebar = ttk.Frame(root, width=280)
        workspace = ttk.PanedWindow(root, orient=tk.VERTICAL)
        root.add(sidebar, weight=0)
        root.add(workspace, weight=1)

        self._build_settings_sidebar(sidebar)
        self._build_image_panel(workspace)
        self._build_bottom_tabs(workspace)
        ttk.Label(self, textvariable=self.status, anchor=tk.W).pack(fill=tk.X, padx=8, pady=(0, 6))

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(toolbar, text="Open", command=self.open_files).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Open sample", command=self.open_sample).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Output folder", command=self.choose_output_dir).pack(side=tk.LEFT, padx=(6, 0))
        self.run_button = ttk.Button(toolbar, text="Run analysis", command=self.run_analysis)
        self.run_button.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(toolbar, text="Clear results", command=self.clear_results).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(toolbar, text="Metric").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Combobox(toolbar, textvariable=self.mtf_metric, values=("mtf_ny2", "mtf_ny4", "mtf50"), state="readonly", width=10).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=self.status, anchor=tk.E).pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def _build_image_panel(self, parent: ttk.PanedWindow) -> None:
        preview_frame = ttk.LabelFrame(parent, text="Image preview")
        parent.add(preview_frame, weight=4)
        controls = ttk.Frame(preview_frame)
        controls.pack(fill=tk.X, padx=8, pady=(6, 0))
        ttk.Button(controls, text="Zoom +", command=lambda: self.zoom_preview(1.25)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Zoom -", command=lambda: self.zoom_preview(0.8)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(controls, text="Fit", command=self.fit_preview).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(controls, text="Drag to pan. Click near an edge for SFR.", anchor=tk.E).pack(side=tk.RIGHT)

        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.preview = tk.Canvas(canvas_frame, background="#f5f5f5", highlightthickness=1, highlightbackground="#c8c8c8")
        self.preview.grid(row=0, column=0, sticky="nsew")
        x_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.preview.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.preview.yview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.preview.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.preview.create_text(320, 200, text="Open an image to run analysis", fill="#555555", tags="placeholder")
        self.preview.bind("<ButtonPress-1>", self.on_preview_press)
        self.preview.bind("<B1-Motion>", self.on_preview_drag)
        self.preview.bind("<ButtonRelease-1>", self.on_preview_release)
        self.preview.bind("<MouseWheel>", self.on_preview_wheel)
        self.preview.bind("<Configure>", self.on_preview_configure)

    def _build_settings_sidebar(self, parent: ttk.Frame) -> None:
        props = ttk.LabelFrame(parent, text="Image")
        props.pack(fill=tk.X, pady=(0, 8))
        self.input_label = ttk.Label(props, text="Input: none", anchor=tk.W)
        self.input_label.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        self.output_label = ttk.Label(props, text=f"Output: {short_path(self.current_output_root)}", anchor=tk.W)
        self.output_label.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        props.columnconfigure(0, weight=1)

        settings = ttk.LabelFrame(parent, text="Input")
        settings.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(settings, text="Linear gamma (8 bit)", variable=self.linear).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(settings, text="Invert brightness", variable=self.invert).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(settings, text="Single edge / ROI", variable=self.single_roi).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        settings.columnconfigure(1, weight=1)

        measurement = ttk.LabelFrame(parent, text="Measurement")
        measurement.pack(fill=tk.X, pady=(0, 8))
        self._labeled_entry(measurement, "Threshold", self.threshold, 0)
        self._labeled_entry(measurement, "Pixel size (um)", self.pixelsize, 1)
        self._labeled_entry(measurement, "MTF contrast", self.mtf, 2)
        measurement.columnconfigure(1, weight=1)

        outputs = ttk.LabelFrame(parent, text="Outputs")
        outputs.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(outputs, text="Annotated image", variable=self.annotate).grid(row=0, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(outputs, text="CSV tables", variable=self.edges).grid(row=1, column=0, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(outputs, text="MTF heat map", variable=self.heatmap).grid(row=2, column=0, sticky=tk.W, padx=8, pady=3)

        advanced = ttk.LabelFrame(parent, text="Advanced")
        advanced.pack(fill=tk.X, pady=(0, 8))
        self._labeled_entry(advanced, "Threshold window", self.threshold_window, 0)
        ttk.Checkbutton(advanced, text="Extended SFR domain", variable=self.full_sfr).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        ttk.Checkbutton(advanced, text="Reduced SFR smoothing", variable=self.nosmoothing).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        advanced.columnconfigure(1, weight=1)

        raw = ttk.LabelFrame(parent, text="Raw import")
        raw.pack(fill=tk.X)
        raw_enable = ttk.Checkbutton(raw, text="Read as raw pixel stream", variable=self.raw)
        raw_enable.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=8, pady=3)
        self._labeled_entry(raw, "Width", self.raw_width, 1, store=self.raw_widgets)
        self._labeled_entry(raw, "Height", self.raw_height, 2, store=self.raw_widgets)
        dtype_label = ttk.Label(raw, text="Data type")
        dtype_label.grid(row=3, column=0, sticky=tk.W, padx=8, pady=3)
        dtype_box = ttk.Combobox(raw, textvariable=self.raw_dtype, values=("uint8", "uint16", "int16", "float32", "float64"), state="readonly")
        dtype_box.grid(row=3, column=1, sticky="ew", padx=8, pady=3)
        order_label = ttk.Label(raw, text="Byte order")
        order_label.grid(row=4, column=0, sticky=tk.W, padx=8, pady=3)
        order_box = ttk.Combobox(raw, textvariable=self.raw_byte_order, values=("little", "big", "native"), state="readonly")
        order_box.grid(row=4, column=1, sticky="ew", padx=8, pady=3)
        self.raw_widgets.extend([dtype_label, dtype_box, order_label, order_box])
        self._labeled_entry(raw, "Header bytes", self.raw_header, 5, store=self.raw_widgets)
        self._labeled_entry(raw, "Channels", self.raw_channels, 6, store=self.raw_widgets)
        raw.columnconfigure(1, weight=1)

    def _build_bottom_tabs(self, parent: ttk.PanedWindow) -> None:
        self.bottom_tabs = ttk.Notebook(parent)
        parent.add(self.bottom_tabs, weight=1)

        results_tab = ttk.Frame(self.bottom_tabs)
        self.result_tree = ttk.Treeview(results_tab, columns=("path",), show="tree", height=7)
        self.result_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.result_tree.bind("<<TreeviewSelect>>", self.on_result_selected)
        result_buttons = ttk.Frame(results_tab)
        result_buttons.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(result_buttons, text="Open selected", command=self.open_selected_result).pack(side=tk.LEFT)
        self.bottom_tabs.add(results_tab, text="Results")

        sfr_tab = ttk.Frame(self.bottom_tabs)
        ttk.Label(sfr_tab, textvariable=self.selected_edge, anchor=tk.W).pack(fill=tk.X, padx=8, pady=(8, 0))
        self.sfr_canvas = tk.Canvas(sfr_tab, background="white", highlightthickness=1, highlightbackground="#c8c8c8")
        self.sfr_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.sfr_canvas.bind("<Configure>", lambda _event: self.redraw_selected_sfr())
        self.sfr_canvas.bind("<Motion>", self.on_sfr_hover)
        self.sfr_canvas.bind("<Leave>", self.on_sfr_leave)
        self.bottom_tabs.add(sfr_tab, text="SFR Curve")

        log_tab = ttk.Frame(self.bottom_tabs)
        self.log_text = ScrolledText(log_tab, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.log_text.config(state=tk.DISABLED)
        self.bottom_tabs.add(log_tab, text="Log")

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

    def open_files(self, auto_run: bool = True) -> None:
        filenames = filedialog.askopenfilenames(title="Select input files", filetypes=SUPPORTED_IMAGE_TYPES)
        if not filenames:
            return
        self.input_files = [Path(name) for name in filenames]
        self.single_roi.set(False)
        self.input_label.config(text=f"Input: {len(self.input_files)} file(s), {self.input_files[0].name}")
        self.status.set("Image loaded; analysis will start")
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
        self.linear.set(True)
        self.single_roi.set(False)
        self.raw.set(False)
        self.input_label.config(text=f"Input: sample, {SAMPLE_CHART.name}")
        self.status.set("Sample loaded; analysis will start")
        self.log(f"Loaded sample chart: {SAMPLE_CHART}")
        self.after(50, self.run_analysis)

    def choose_output_dir(self) -> None:
        dirname = filedialog.askdirectory(title="Select output directory")
        if not dirname:
            return
        self.current_output_root = Path(dirname)
        self.output_label.config(text=f"Output: {short_path(self.current_output_root)}")
        self.log(f"Output folder set to: {self.current_output_root}")

    def _update_raw_controls(self) -> None:
        state = tk.NORMAL if self.raw.get() else tk.DISABLED
        for widget in self.raw_widgets:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def log(self, message: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def current_values(self) -> dict[str, object]:
        return {
            "threshold": self.threshold.get(),
            "threshold_window": self.threshold_window.get(),
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
            "pixelsize": self.pixelsize.get(),
            "raw": self.raw.get(),
            "raw_width": self.raw_width.get(),
            "raw_height": self.raw_height.get(),
            "raw_dtype": self.raw_dtype.get(),
            "raw_byte_order": self.raw_byte_order.get(),
            "raw_header": self.raw_header.get(),
            "raw_channels": self.raw_channels.get(),
            "log_level": "INFO",
        }

    def run_analysis(self) -> None:
        if not self.input_files:
            messagebox.showwarning("No input", "Select at least one input image first.")
            return
        if not self.annotate.get() and not self.edges.get() and not self.heatmap.get():
            messagebox.showwarning("No outputs", "Select at least one output.")
            return
        self.run_button.config(state=tk.DISABLED)
        self.status.set("Analyzing...")
        self.log(f"Starting analysis for {len(self.input_files)} file(s)")
        values = self.current_values()
        thread = threading.Thread(target=self._run_worker, args=(self.input_files.copy(), values), daemon=True)
        thread.start()

    def _run_worker(self, files: list[Path], values: dict[str, object]) -> None:
        try:
            for input_path in files:
                output_dir = self.current_output_root / input_path.stem
                args = namespace_from_gui_values(input_path, output_dir, values)
                lum, annotated, measurements = mtf_mapper_py.analyze_image(args)
                if args.edges:
                    mtf_mapper_py.write_edge_tables(output_dir, measurements)
                if args.annotate:
                    mtf_mapper_py.write_annotation(output_dir, annotated)
                if args.heatmap:
                    mtf_mapper_py.write_heatmap(output_dir, lum, measurements)
                self.worker_queue.put(("result", GuiRunResult(input_path, output_dir, len(measurements), measurements)))
            self.worker_queue.put(("done", None))
        except Exception as exc:
            self.worker_queue.put(("error", exc))

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "result":
                    self.add_result(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    self.run_button.config(state=tk.NORMAL)
                    self.status.set("Analysis failed")
                    self.log(f"Analysis failed: {payload}")
                    messagebox.showerror("Analysis failed", str(payload))
                elif kind == "done":
                    self.run_button.config(state=tk.NORMAL)
                    self.status.set("Analysis complete")
                    self.log("Analysis complete")
        except queue.Empty:
            pass
        self.after(100, self._poll_worker_queue)

    def add_result(self, result: GuiRunResult) -> None:
        parent_id = self.result_tree.insert("", tk.END, text=f"{result.input_path.name} ({result.edge_count} edges)")
        self.result_measurements[result.output_dir] = result.measurements
        for path in sorted(result.output_dir.glob("*")):
            item_id = self.result_tree.insert(parent_id, tk.END, text=path.name)
            self.result_rows[item_id] = path
        self.result_tree.item(parent_id, open=True)
        self.log(f"{result.input_path.name}: measured {result.edge_count} edges")
        annotated = result.output_dir / "annotated.png"
        if annotated.exists():
            self.show_image(annotated, result.measurements)
        else:
            heatmap = result.output_dir / "mtf_heatmap.png"
            if heatmap.exists():
                self.show_image(heatmap, result.measurements)

    def clear_results(self) -> None:
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_rows.clear()
        self.result_measurements.clear()
        self.clear_preview("Open an image to run analysis")
        self.preview_image = None
        self.preview_state = None
        self.preview_path = None
        self.preview_measurements = []
        self.selected_measurement = None
        self.sfr_plot_state = None
        self.selected_edge.set("No edge selected")
        self.sfr_canvas.delete("all")
        self.status.set("Results cleared")
        self.log("Results cleared")

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

    def clear_preview(self, text: str) -> None:
        self.preview.delete("all")
        self.preview.configure(scrollregion=(0, 0, max(self.preview.winfo_width(), 320), max(self.preview.winfo_height(), 200)))
        self.preview.create_text(
            max(self.preview.winfo_width() // 2, 160),
            max(self.preview.winfo_height() // 2, 100),
            text=text,
            fill="#555555",
            tags="placeholder",
        )

    def calculate_fit_scale(self, path: Path) -> float:
        image_width, image_height = image_size_from_cv(path)
        canvas_width = max(self.preview.winfo_width() - 18, 320)
        canvas_height = max(self.preview.winfo_height() - 18, 220)
        return max(0.05, min(canvas_width / image_width, canvas_height / image_height, 1.0))

    def show_image(self, path: Path, measurements: list[mtf_mapper_py.EdgeMeasurement] | None = None) -> None:
        self.preview_path = path
        self.preview_measurements = measurements or []
        try:
            self.preview_fit_scale = self.calculate_fit_scale(path)
        except ValueError:
            self.preview_fit_scale = 1.0
        self.preview_zoom = self.preview_fit_scale
        self.render_preview(reset_view=True)

    def render_preview(self, reset_view: bool = False) -> None:
        if self.preview_path is None:
            self.clear_preview("Open an image to run analysis")
            return
        try:
            image, image_width, image_height, display_width, display_height = photo_image_from_cv(
                self.preview_path,
                self.preview_zoom,
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
            self.preview.create_image(offset_x, offset_y, image=image, anchor=tk.NW)
            self.preview.configure(scrollregion=(0, 0, scroll_width, scroll_height))
            if reset_view:
                self.preview.xview_moveto(0.0)
                self.preview.yview_moveto(0.0)
            hint = " - click near an edge to view SFR" if self.preview_measurements else ""
            self.status.set(f"Previewing {self.preview_path.name}{hint}")
        except (tk.TclError, ValueError) as exc:
            self.clear_preview(f"Cannot preview {self.preview_path.name}")
            self.log(str(exc))
            self.preview_image = None
            self.preview_state = None

    def fit_preview(self) -> None:
        if self.preview_path is None:
            return
        self.preview_fit_scale = self.calculate_fit_scale(self.preview_path)
        self.preview_zoom = self.preview_fit_scale
        self.render_preview(reset_view=True)

    def zoom_preview(self, factor: float) -> None:
        if self.preview_path is None:
            return
        old_x = self.preview.xview()[0]
        old_y = self.preview.yview()[0]
        self.preview_zoom = min(max(self.preview_zoom * factor, 0.05), 8.0)
        self.render_preview(reset_view=False)
        self.preview.xview_moveto(old_x)
        self.preview.yview_moveto(old_y)

    def on_preview_wheel(self, event: tk.Event) -> None:
        if self.preview_path is None:
            return
        self.zoom_preview(1.15 if event.delta > 0 else 1 / 1.15)

    def on_preview_configure(self, _event: tk.Event) -> None:
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
        if math.hypot(event.x - start_x, event.y - start_y) > 6:
            return
        self.on_preview_click(event)

    def on_preview_click(self, event: tk.Event) -> None:
        state = self.preview_state
        if state is None or not state.measurements:
            return
        canvas_x = int(self.preview.canvasx(event.x))
        canvas_y = int(self.preview.canvasy(event.y))
        image_x, image_y = preview_to_image_coords(canvas_x, canvas_y, state)
        if image_x < 0 or image_y < 0:
            return
        if image_x > state.image_width or image_y > state.image_height:
            return
        measurement = nearest_measurement(state.measurements, image_x, image_y)
        if measurement is None:
            return
        distance = distance_to_measurement(measurement, image_x, image_y)
        if distance > 72.0:
            self.status.set("Click closer to an annotated edge to view its SFR curve")
            return
        self.show_sfr_curve(measurement)

    def show_sfr_curve(self, measurement: mtf_mapper_py.EdgeMeasurement) -> None:
        self.selected_measurement = measurement
        self.selected_edge.set(
            f"Block {measurement.block_id}  Edge ({measurement.edge_x:.1f}, {measurement.edge_y:.1f})  "
            f"{measurement.mtf_column}={measurement.mtf_value:.4g}"
        )
        self.redraw_selected_sfr()
        self.bottom_tabs.select(self.sfr_canvas.master)
        self.status.set("Showing SFR curve for selected edge")

    def redraw_selected_sfr(self) -> None:
        self.sfr_canvas.delete("all")
        if self.selected_measurement is None:
            self.sfr_plot_state = None
            self.sfr_canvas.create_text(
                max(self.sfr_canvas.winfo_width() // 2, 160),
                max(self.sfr_canvas.winfo_height() // 2, 100),
                text="Click an annotated edge to view its SFR curve",
                fill="#555555",
            )
            return
        self.sfr_plot_state = draw_sfr_curve(self.sfr_canvas, self.selected_measurement)

    def on_sfr_hover(self, event: tk.Event) -> None:
        state = self.sfr_plot_state
        if state is None:
            return
        self.sfr_canvas.delete("hover")
        if event.x < state.x0 or event.x > state.x1 or event.y < state.y1 or event.y > state.y0:
            return
        frac = (event.x - state.x0) / max(state.x1 - state.x0, 1.0)
        frequency = frac * state.max_frequency
        sample_index = min(max(frequency * 64.0, 0.0), len(state.measurement.sfr) - 1)
        indices = list(range(len(state.measurement.sfr)))
        value = float(np.interp(sample_index, indices, state.measurement.sfr))
        y = state.y0 - min(max(value, 0.0), state.y_max) / state.y_max * (state.y0 - state.y1)
        self.sfr_canvas.create_line(event.x, state.y1, event.x, state.y0, fill="#d12b2b", dash=(5, 4), tags="hover")
        self.sfr_canvas.create_oval(event.x - 3, y - 3, event.x + 3, y + 3, fill="#d12b2b", outline="", tags="hover")
        label = f"x={frequency:.4f} c/p, y={value:.4f}"
        label_x = min(max(event.x + 10, state.x0 + 4), state.x1 - 135)
        label_y = state.y1 + 10
        self.sfr_canvas.create_rectangle(label_x - 4, label_y - 4, label_x + 132, label_y + 18, fill="white", outline="#b0b0b0", tags="hover")
        self.sfr_canvas.create_text(label_x, label_y, text=label, anchor=tk.NW, fill="#333333", tags="hover")

    def on_sfr_leave(self, _event: tk.Event) -> None:
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
