#!/usr/bin/env python3
"""Core Python port of the mtf_mapper analyzer.

This is intentionally smaller than the C++ application: it focuses on
automatic rectangular target detection, single-ROI slanted edges, MTF/SFR
estimation, and the most useful tabular/annotated outputs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised in dependency-poor envs
    cv2 = None

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - fallback covered by tests
    savgol_filter = None


LOGGER = logging.getLogger("mtf_mapper_py")
MTF_NYQUIST_CP = 0.5
MTF_FIXED_FREQUENCIES = {
    "mtf_ny2": MTF_NYQUIST_CP / 2.0,
    "mtf_ny4": MTF_NYQUIST_CP / 4.0,
}


@dataclass
class EdgeMeasurement:
    block_id: int
    edge_x: float
    edge_y: float
    mtf_value: float
    mtf_metric: str
    mtf_column: str
    corner_x: float
    corner_y: float
    edge_angle: float
    radial_angle: float
    sfr: np.ndarray
    quality: float
    edge_start_x: float = 0.0
    edge_start_y: float = 0.0
    edge_end_x: float = 0.0
    edge_end_y: float = 0.0


@dataclass
class HeatmapBlock:
    block_id: int
    x: float
    y: float
    mtf_value: float
    mtf_metric: str
    mtf_column: str


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for image processing. Install dependencies with "
            "`uv sync` or `python -m pip install -r requirements.txt`."
        )


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(3, int(width) | 1)
    if values.size < width:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def smooth_esf(values: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled or values.size < 9:
        return values
    window = min(values.size if values.size % 2 == 1 else values.size - 1, 21)
    if window < 5:
        return values
    if savgol_filter is not None:
        return savgol_filter(values, window_length=window, polyorder=3, mode="interp")
    return moving_average(values, window)


def dtype_for_raw(raw_dtype: str, byte_order: str) -> np.dtype:
    dtype = np.dtype(raw_dtype)
    if dtype.itemsize == 1 or byte_order == "native":
        return dtype
    endian = "<" if byte_order == "little" else ">"
    return dtype.newbyteorder(endian)


def load_raw_image(
    path: Path,
    width: int,
    height: int,
    raw_dtype: str,
    byte_order: str,
    header: int,
    channels: int,
) -> np.ndarray:
    dtype = dtype_for_raw(raw_dtype, byte_order)
    expected_values = width * height * channels
    with path.open("rb") as fin:
        fin.seek(header)
        data = np.fromfile(fin, dtype=dtype, count=expected_values)
    if data.size != expected_values:
        raise ValueError(
            f"raw input ended early; expected {expected_values} values after {header} header bytes, "
            f"read {data.size}"
        )
    if channels == 1:
        return data.reshape((height, width))
    return data.reshape((height, width, channels))


def normalize_image(img: np.ndarray) -> np.ndarray:
    if img.dtype in (np.uint8, np.uint16):
        return img.astype(np.float64) / np.iinfo(img.dtype).max
    if np.issubdtype(img.dtype, np.signedinteger):
        info = np.iinfo(img.dtype)
        return (img.astype(np.float64) - info.min) / (info.max - info.min)
    if np.issubdtype(img.dtype, np.floating):
        arr = img.astype(np.float64)
        finite = np.isfinite(arr)
        if not finite.any():
            raise ValueError("raw floating-point image does not contain finite values")
        min_val = float(np.nanmin(arr[finite]))
        max_val = float(np.nanmax(arr[finite]))
        if max_val <= min_val:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - min_val) / (max_val - min_val)
    raise ValueError(f"unsupported image dtype {img.dtype}")


def display_copy(img: np.ndarray) -> np.ndarray:
    arr = normalize_image(img)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def luminance_from_array(
    img: np.ndarray,
    linear: bool,
    invert: bool,
    apply_srgb_for_uint8: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if img.dtype not in (np.uint8, np.uint16) and not (
        np.issubdtype(img.dtype, np.signedinteger) or np.issubdtype(img.dtype, np.floating)
    ):
        raise ValueError("invalid image type; numeric 8-bit, 16-bit, signed integer, or float images are supported")

    original = img.copy()
    arr = normalize_image(img)
    if arr.ndim == 2:
        lum = arr
    else:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        # OpenCV loads color images as BGR.
        bgr = arr
        if apply_srgb_for_uint8 and img.dtype == np.uint8 and not linear:
            bgr = srgb_to_linear(bgr)
        lum = 0.0722 * bgr[:, :, 0] + 0.7152 * bgr[:, :, 1] + 0.2126 * bgr[:, :, 2]

    if apply_srgb_for_uint8 and img.dtype == np.uint8 and img.ndim == 2 and not linear:
        lum = srgb_to_linear(lum)
    if invert:
        lum = 1.0 - lum
    return np.clip(lum, 0.0, 1.0), original


def load_luminance(path: Path, linear: bool, invert: bool) -> tuple[np.ndarray, np.ndarray]:
    require_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not open input image <{path}>")
    if img.dtype not in (np.uint8, np.uint16):
        raise ValueError("invalid image type; only 8-bit and 16-bit unsigned images are supported")
    return luminance_from_array(img, linear=linear, invert=invert, apply_srgb_for_uint8=True)


def luminance_to_bgr(lum: np.ndarray) -> np.ndarray:
    require_cv2()
    img8 = np.clip(lum * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)


def odd_window(size: int) -> int:
    size = max(15, int(size))
    if size % 2 == 0:
        size += 1
    return size


def threshold_dark_objects(lum: np.ndarray, threshold: float, threshold_window: float) -> np.ndarray:
    require_cv2()
    img8 = np.clip(lum * 255.0, 0, 255).astype(np.uint8)
    win = odd_window(round(min(lum.shape[:2]) * threshold_window))
    adaptive = cv2.adaptiveThreshold(
        img8,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        win,
        3,
    )
    global_mask = (lum < threshold).astype(np.uint8) * 255
    mask = cv2.bitwise_and(adaptive, global_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def order_box_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def detect_boxes(lum: np.ndarray, threshold: float, threshold_window: float) -> list[np.ndarray]:
    require_cv2()
    mask = threshold_dark_objects(lum, threshold, threshold_window)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = lum.shape[0] * lum.shape[1]
    boxes: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(60.0, image_area * 0.00005):
            continue
        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        if min(width, height) < 8:
            continue
        rectangularity = area / max(width * height, 1.0)
        if rectangularity < 0.55:
            LOGGER.warning("skipping low-rectangularity contour with score %.2f", rectangularity)
            continue
        boxes.append(order_box_points(cv2.boxPoints(rect).astype(np.float64)))
    boxes.sort(key=lambda box: (box[:, 1].mean(), box[:, 0].mean()))
    return boxes


def sample_bilinear(lum: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    require_cv2()
    map_x = x.astype(np.float32)
    map_y = y.astype(np.float32)
    return cv2.remap(
        lum.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float64)


def esf_from_edge(
    lum: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    oversampling: int = 8,
    radius: float = 12.0,
) -> tuple[np.ndarray, float, float]:
    tangent = p1 - p0
    length = float(np.linalg.norm(tangent))
    if length < 8:
        raise ValueError("edge is too short")
    tangent /= length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    center = (p0 + p1) * 0.5

    samples_along = max(12, int(length) - 4)
    along = np.linspace(-0.45 * length, 0.45 * length, samples_along)
    across = np.arange(-radius, radius + 1.0 / oversampling, 1.0 / oversampling)
    xs = center[0] + along[:, None] * tangent[0] + across[None, :] * normal[0]
    ys = center[1] + along[:, None] * tangent[1] + across[None, :] * normal[1]

    inside = (xs >= 0) & (ys >= 0) & (xs < lum.shape[1] - 1) & (ys < lum.shape[0] - 1)
    if inside.mean() < 0.65:
        raise ValueError("edge sampling window falls outside image")

    values = sample_bilinear(lum, xs, ys)
    values = np.where(inside, values, np.nan)
    esf = np.nanmean(values, axis=0)
    valid = np.isfinite(esf)
    if valid.sum() < oversampling * 8:
        raise ValueError("not enough valid ESF samples")
    esf = np.interp(np.arange(esf.size), np.flatnonzero(valid), esf[valid])

    low_mean = float(np.mean(esf[:oversampling * 4]))
    high_mean = float(np.mean(esf[-oversampling * 4:]))
    if low_mean > high_mean:
        esf = esf[::-1]
    contrast = abs(high_mean - low_mean)
    return esf, 1.0 / oversampling, contrast


def sfr_from_esf(esf: np.ndarray, sample_spacing: float, smooth: bool, full_sfr: bool) -> tuple[np.ndarray, np.ndarray]:
    esf = smooth_esf(esf.astype(np.float64), smooth)
    lsf = np.gradient(esf, sample_spacing)
    lsf -= np.mean(lsf)
    if lsf.size > 1:
        lsf *= np.hamming(lsf.size)
    response = np.abs(np.fft.rfft(lsf))
    if response.size == 0 or response[0] == 0:
        raise ValueError("empty SFR response")
    sfr = response / response[0]
    freqs = np.fft.rfftfreq(lsf.size, d=sample_spacing)
    max_freq = 2.0 if full_sfr else 1.0
    keep = freqs <= max_freq
    return freqs[keep], sfr[keep]


def interpolate_mtf(freqs: np.ndarray, sfr: np.ndarray, contrast: float) -> float:
    if not 0 < contrast < 1:
        raise ValueError("MTF contrast must be between 1 and 99")
    for idx in range(1, len(sfr)):
        if sfr[idx] <= contrast <= sfr[idx - 1]:
            x0, x1 = freqs[idx - 1], freqs[idx]
            y0, y1 = sfr[idx - 1], sfr[idx]
            if abs(y1 - y0) < 1e-12:
                return float(x1)
            return float(x0 + (contrast - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def mtf_value_at_frequency(freqs: np.ndarray, sfr: np.ndarray, frequency: float) -> float:
    if frequency < freqs[0] or frequency > freqs[-1]:
        return float("nan")
    return float(np.interp(frequency, freqs, sfr))


def reported_mtf_value(
    freqs: np.ndarray,
    sfr: np.ndarray,
    metric: str,
    mtf_contrast: float,
    pixel_size: float | None,
) -> float:
    if metric == "mtf50":
        value = interpolate_mtf(freqs, sfr, mtf_contrast / 100.0)
        if pixel_size is not None and math.isfinite(value):
            value *= 1000.0 / pixel_size
        return value
    return mtf_value_at_frequency(freqs, sfr, MTF_FIXED_FREQUENCIES[metric])


def mtf_metric_column(metric: str, pixel_size: float | None) -> str:
    if metric == "mtf50":
        return "mtf50_lpmm" if pixel_size is not None else "mtf50_cp"
    return metric


def resample_sfr(freqs: np.ndarray, sfr: np.ndarray, full_sfr: bool) -> np.ndarray:
    count = 128 if full_sfr else 64
    target_freqs = np.arange(count, dtype=np.float64) / 64.0
    return np.interp(target_freqs, freqs, sfr, left=sfr[0], right=sfr[-1])


def folded_edge_angle(p0: np.ndarray, p1: np.ndarray) -> float:
    angle = math.degrees(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0])))
    angle = abs(angle) % 90.0
    return 90.0 - angle if angle > 45.0 else angle


def radial_angle(point: np.ndarray, shape: tuple[int, int]) -> float:
    center = np.array([shape[1] / 2.0, shape[0] / 2.0], dtype=np.float64)
    delta = point - center
    return math.degrees(math.atan2(float(delta[1]), float(delta[0])))


def measure_edge(
    lum: np.ndarray,
    block_id: int,
    p0: np.ndarray,
    p1: np.ndarray,
    corner: np.ndarray,
    mtf_contrast: float,
    mtf_metric: str,
    full_sfr: bool,
    smooth: bool,
    pixel_size: float | None,
) -> EdgeMeasurement:
    esf, spacing, edge_contrast = esf_from_edge(lum, p0, p1)
    freqs, sfr = sfr_from_esf(esf, spacing, smooth=smooth, full_sfr=full_sfr)
    mtf_value = reported_mtf_value(freqs, sfr, mtf_metric, mtf_contrast, pixel_size)
    center = (p0 + p1) * 0.5
    return EdgeMeasurement(
        block_id=block_id,
        edge_x=float(center[0]),
        edge_y=float(center[1]),
        mtf_value=mtf_value,
        mtf_metric=mtf_metric,
        mtf_column=mtf_metric_column(mtf_metric, pixel_size),
        corner_x=float(corner[0]),
        corner_y=float(corner[1]),
        edge_angle=folded_edge_angle(p0, p1),
        radial_angle=radial_angle(center, lum.shape),
        sfr=resample_sfr(freqs, sfr, full_sfr),
        quality=edge_contrast,
        edge_start_x=float(p0[0]),
        edge_start_y=float(p0[1]),
        edge_end_x=float(p1[0]),
        edge_end_y=float(p1[1]),
    )


def box_edges(box: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for idx in range(4):
        yield box[idx], box[(idx + 1) % 4], box[idx]


def detect_single_roi_box(lum: np.ndarray, threshold: float) -> np.ndarray:
    mask = (lum < threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no edge-like object found in single ROI")
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    return order_box_points(cv2.boxPoints(rect).astype(np.float64))


def analyze_image(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[EdgeMeasurement]]:
    input_path = Path(args.input_image)
    if args.raw:
        raw_img = load_raw_image(
            input_path,
            width=args.raw_width,
            height=args.raw_height,
            raw_dtype=args.raw_dtype,
            byte_order=args.raw_byte_order,
            header=args.raw_header,
            channels=args.raw_channels,
        )
        # ImageJ-style raw imports are scientific pixel data, so treat them as linear intensity.
        lum, original = luminance_from_array(raw_img, linear=True, invert=args.invert, apply_srgb_for_uint8=False)
    else:
        lum, original = load_luminance(input_path, linear=args.linear, invert=args.invert)
    boxes = [detect_single_roi_box(lum, args.threshold)] if args.single_roi else detect_boxes(
        lum, args.threshold, args.threshold_window
    )
    if not boxes:
        raise ValueError("no dark rectangular objects found; try a different --threshold")

    measurements: list[EdgeMeasurement] = []
    for block_id, box in enumerate(boxes, start=1):
        for p0, p1, corner in box_edges(box):
            try:
                measurement = measure_edge(
                    lum,
                    block_id,
                    p0,
                    p1,
                    corner,
                    mtf_contrast=args.mtf,
                    mtf_metric=args.mtf_metric,
                    full_sfr=args.full_sfr,
                    smooth=not args.nosmoothing,
                    pixel_size=args.pixelsize,
                )
            except ValueError as exc:
                LOGGER.warning("skipping edge in block %d: %s", block_id, exc)
                continue
            if measurement.quality < 0.05:
                LOGGER.warning(
                    "low-confidence edge at %.1f %.1f has weak contrast %.3f",
                    measurement.edge_x,
                    measurement.edge_y,
                    measurement.quality,
                )
            measurements.append(measurement)

    if not measurements:
        raise ValueError("targets were found, but no usable edges could be measured")
    return lum, make_annotation(lum, original, measurements), measurements


def measure_image(args: argparse.Namespace) -> tuple[np.ndarray, list[EdgeMeasurement]]:
    _lum, annotated, measurements = analyze_image(args)
    return annotated, measurements


def make_annotation(lum: np.ndarray, original: np.ndarray, measurements: Sequence[EdgeMeasurement]) -> np.ndarray:
    require_cv2()
    if original.ndim == 2:
        annotated = luminance_to_bgr(lum)
    else:
        base = display_copy(original[:, :, :3])
        annotated = base.copy()
    for m in measurements:
        color = (0, 0, 255)
        if not math.isfinite(m.mtf_value):
            label = "N/A"
        else:
            label = f"{m.mtf_value:.1f}" if m.mtf_value >= 10 else f"{m.mtf_value:.3f}"
        pos = (int(round(m.edge_x)), int(round(m.edge_y)))
        cv2.circle(annotated, pos, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, pos, 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(annotated, pos, 3, color, -1, cv2.LINE_AA)
        text_pos = (pos[0] + 6, pos[1] - 6)
        cv2.putText(annotated, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(annotated, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return annotated


def write_edge_tables(output_dir: Path, measurements: Sequence[EdgeMeasurement]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mtf_path = output_dir / "edge_mtf_values.csv"
    sfr_path = output_dir / "edge_sfr_values.csv"
    with mtf_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        metric_column = measurements[0].mtf_column if measurements else "mtf_ny4"
        writer.writerow(["block_id", "edge_x", "edge_y", metric_column, "corner_x", "corner_y"])
        for m in measurements:
            writer.writerow(
                [
                    m.block_id,
                    f"{m.edge_x:.6f}",
                    f"{m.edge_y:.6f}",
                    f"{m.mtf_value:.9g}",
                    f"{m.corner_x:.6f}",
                    f"{m.corner_y:.6f}",
                ]
            )
    with sfr_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        sfr_count = len(measurements[0].sfr) if measurements else 0
        writer.writerow(
            ["block_id", "edge_x", "edge_y", "edge_angle", "radial_angle"]
            + [f"sfr_{idx:03d}" for idx in range(sfr_count)]
        )
        for m in measurements:
            writer.writerow(
                [
                    m.block_id,
                    f"{m.edge_x:.6f}",
                    f"{m.edge_y:.6f}",
                    f"{m.edge_angle:.6f}",
                    f"{m.radial_angle:.6f}",
                ]
                + [f"{v:.9g}" for v in m.sfr]
            )


def write_annotation(output_dir: Path, annotated: np.ndarray) -> None:
    require_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "annotated.png"
    if not cv2.imwrite(str(out_path), annotated):
        raise ValueError(f"could not write {out_path}")


def mtf_heatmap_limits(values_source: Sequence[HeatmapBlock | EdgeMeasurement]) -> tuple[float, float]:
    values = np.array([m.mtf_value for m in values_source if math.isfinite(m.mtf_value)], dtype=np.float64)
    if values.size == 0:
        return 0.0, 1.0
    if values_source[0].mtf_metric in MTF_FIXED_FREQUENCIES:
        return 0.0, max(1.0, float(np.nanpercentile(values, 95)))
    low = float(np.nanpercentile(values, 5))
    high = float(np.nanpercentile(values, 95))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if high <= low:
        high = low + 1.0
    return low, high


def heatmap_blocks_from_measurements(measurements: Sequence[EdgeMeasurement]) -> list[HeatmapBlock]:
    grouped: dict[int, list[EdgeMeasurement]] = {}
    for measurement in measurements:
        if math.isfinite(measurement.mtf_value):
            grouped.setdefault(measurement.block_id, []).append(measurement)
    blocks: list[HeatmapBlock] = []
    for block_id, edges in sorted(grouped.items()):
        blocks.append(
            HeatmapBlock(
                block_id=block_id,
                x=float(np.mean([edge.edge_x for edge in edges])),
                y=float(np.mean([edge.edge_y for edge in edges])),
                mtf_value=float(np.mean([edge.mtf_value for edge in edges])),
                mtf_metric=edges[0].mtf_metric,
                mtf_column=edges[0].mtf_column,
            )
        )
    return blocks


def make_mtf_heatmap(lum: np.ndarray, measurements: Sequence[EdgeMeasurement]) -> np.ndarray:
    require_cv2()
    blocks = heatmap_blocks_from_measurements(measurements)
    if not blocks:
        raise ValueError("cannot build MTF heat map without finite block-average MTF measurements")

    height, width = lum.shape[:2]
    max_grid = 640
    scale = min(1.0, max_grid / max(width, height))
    grid_w = max(2, int(round(width * scale)))
    grid_h = max(2, int(round(height * scale)))

    points = np.array([[block.x * scale, block.y * scale] for block in blocks], dtype=np.float64)
    values = np.array([block.mtf_value for block in blocks], dtype=np.float64)
    xs = np.arange(grid_w, dtype=np.float64)
    ys = np.arange(grid_h, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    estimate = np.zeros((grid_h, grid_w), dtype=np.float64)
    weights = np.zeros_like(estimate)
    for (point_x, point_y), value in zip(points, values):
        distance2 = (grid_x - point_x) ** 2 + (grid_y - point_y) ** 2
        weight = 1.0 / np.maximum(distance2, 1.0)
        estimate += weight * value
        weights += weight
    estimate /= np.maximum(weights, 1e-12)

    vmin, vmax = mtf_heatmap_limits(blocks)
    normalized = np.clip((estimate - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    heat_small = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat_small, (width, height), interpolation=cv2.INTER_CUBIC)
    base = luminance_to_bgr(lum)
    output = cv2.addWeighted(base, 0.28, heat, 0.72, 0.0)

    bar_w = 28
    margin = 18
    bar_h = max(80, min(height - 2 * margin, 240))
    x0 = max(width - margin - bar_w, 0)
    y0 = margin
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    colorbar = cv2.applyColorMap(np.repeat(gradient, bar_w, axis=1), cv2.COLORMAP_JET)
    output[y0 : y0 + bar_h, x0 : x0 + bar_w] = colorbar
    cv2.rectangle(output, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1, cv2.LINE_AA)
    label_color = (255, 255, 255)
    shadow = (0, 0, 0)
    metric = blocks[0].mtf_column
    cv2.putText(output, metric, (margin, margin + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, shadow, 3, cv2.LINE_AA)
    cv2.putText(output, metric, (margin, margin + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, label_color, 1, cv2.LINE_AA)
    for text, yy in ((f"{vmax:.3g}", y0 + 10), (f"{vmin:.3g}", y0 + bar_h)):
        pos = (max(x0 - 56, 0), yy)
        cv2.putText(output, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.42, shadow, 3, cv2.LINE_AA)
        cv2.putText(output, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)
    return output


def write_heatmap(output_dir: Path, lum: np.ndarray, measurements: Sequence[EdgeMeasurement]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "mtf_heatmap.png"
    heatmap = make_mtf_heatmap(lum, measurements)
    if not cv2.imwrite(str(out_path), heatmap):
        raise ValueError(f"could not write {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Core Python MTF Mapper analyzer")
    parser.add_argument("input_image", help="input image file name")
    parser.add_argument("output_dir", help="directory for output files")
    parser.add_argument("-t", "--threshold", type=float, default=0.55, help="dark object threshold in [0,1]")
    parser.add_argument("--threshold-window", type=float, default=1.0 / 3.0, help="adaptive threshold window fraction")
    parser.add_argument("-l", "--linear", action="store_true", help="treat 8-bit input as linear")
    parser.add_argument("--invert", action="store_true", help="invert brightness before processing")
    parser.add_argument("--single-roi", action="store_true", help="treat input as a single cropped edge/target")
    parser.add_argument(
        "--mtf-metric",
        default="mtf_ny4",
        choices=["mtf_ny2", "mtf_ny4", "mtf50"],
        help="reported MTF value: contrast at Nyquist/2, contrast at Nyquist/4, or legacy MTF50 crossing",
    )
    parser.add_argument("--mtf", type=float, default=50.0, help="target contrast percentage used only with --mtf-metric mtf50")
    parser.add_argument("-a", "--annotate", action="store_true", help="write annotated.png")
    parser.add_argument("-q", "--edges", action="store_true", help="write edge_mtf_values.csv and edge_sfr_values.csv")
    parser.add_argument("--heatmap", action="store_true", help="write mtf_heatmap.png with field-wide MTF distribution")
    parser.add_argument("--full-sfr", action="store_true", help="write SFR samples up to 2 cycles/pixel")
    parser.add_argument("--nosmoothing", action="store_true", help="disable ESF smoothing")
    parser.add_argument("--pixelsize", type=float, help="pixel pitch in microns; output MTF in lp/mm")
    parser.add_argument("--raw", action="store_true", help="read input as an ImageJ-style headerless raw pixel stream")
    parser.add_argument("--raw-width", type=int, help="raw image width in pixels")
    parser.add_argument("--raw-height", type=int, help="raw image height in pixels")
    parser.add_argument(
        "--raw-dtype",
        default="uint16",
        choices=["uint8", "uint16", "int16", "float32", "float64"],
        help="raw sample type",
    )
    parser.add_argument(
        "--raw-byte-order",
        default="little",
        choices=["little", "big", "native"],
        help="byte order for multi-byte raw sample types",
    )
    parser.add_argument("--raw-header", type=int, default=0, help="number of header bytes to skip before pixel data")
    parser.add_argument("--raw-channels", type=int, default=1, choices=[1, 3, 4], help="number of interleaved raw channels")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be in the open interval (0, 1)")
    if not 0.0 < args.threshold_window <= 1.0:
        parser.error("--threshold-window must be in the interval (0, 1]")
    if not 1.0 <= args.mtf <= 99.0:
        parser.error("--mtf must be in the interval [1, 99]")
    if args.pixelsize is not None and args.pixelsize <= 0:
        parser.error("--pixelsize must be positive")
    if args.raw:
        if args.raw_width is None or args.raw_height is None:
            parser.error("--raw requires --raw-width and --raw-height")
        if args.raw_width <= 0 or args.raw_height <= 0:
            parser.error("--raw-width and --raw-height must be positive")
        if args.raw_header < 0:
            parser.error("--raw-header must be zero or positive")
    elif any(value is not None for value in (args.raw_width, args.raw_height)) or args.raw_header != 0 or args.raw_channels != 1:
        parser.error("raw metadata options require --raw")
    if not args.annotate and not args.edges and not args.heatmap:
        args.annotate = True
        args.edges = True
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    try:
        lum, annotated, measurements = analyze_image(args)
        output_dir = Path(args.output_dir)
        if args.edges:
            write_edge_tables(output_dir, measurements)
        if args.annotate:
            write_annotation(output_dir, annotated)
        if args.heatmap:
            write_heatmap(output_dir, lum, measurements)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("measured %d edges", len(measurements))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
