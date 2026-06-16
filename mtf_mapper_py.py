#!/usr/bin/env python3
"""Core Python port of the mtf_mapper analyzer.

This is intentionally smaller than the C++ application: it focuses on
automatic rectangular target detection, single-ROI slanted edges, MTF/SFR
estimation, and the most useful tabular/annotated outputs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import math
import sys
from dataclasses import dataclass, field
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
ANNOTATION_COLOR_BGR = (255, 0, 255)
THRESHOLD_MODE_ALIASES = {
    "Hybrid (adaptive + global)": "hybrid",
    "Adaptive only": "adaptive",
    "Global only": "global",
}
ESF_METHOD_ALIASES = {
    "Pixel binning": "pixel-binned",
    "Auto fallback": "auto",
    "Interpolated profiles": "interpolated",
}
RAW_NORMALIZATION_ALIASES = {
    "Auto levels": "auto",
    "Bit depth": "bit-depth",
    "Manual levels": "manual",
    "Full dtype range": "dtype-range",
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
    esf: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    lsf: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    sample_spacing: float = 1.0
    quality_score: float = 0.0
    quality_label: str = "Unknown"
    quality_notes: tuple[str, ...] = ()
    esf_method: str = "pixel-binned"
    bin_occupancy: float = 1.0


@dataclass
class EsfResult:
    values: np.ndarray
    sample_spacing: float
    contrast: float
    method: str
    bin_occupancy: float = 1.0


@dataclass
class DetectionReport:
    threshold_mode: str
    threshold: float
    threshold_window: float
    contour_count: int = 0
    accepted_count: int = 0
    rejected_small_area: int = 0
    rejected_short_side: int = 0
    rejected_shape: int = 0
    rejected_fiducial: int = 0
    fiducial_filter_ratio: float = 0.0

    def suggestions(self) -> list[str]:
        if self.threshold_mode == "manual":
            return ["Using manually selected target regions."]
        suggestions: list[str] = []
        if self.contour_count == 0:
            suggestions.append("Try Adaptive only for uneven illumination or adjust the target threshold.")
        if self.rejected_small_area:
            suggestions.append("Targets may be too small; use a higher-resolution image or crop closer.")
        if self.rejected_shape:
            suggestions.append("Some candidates were not rectangular; improve target contrast or framing.")
        if self.rejected_fiducial:
            suggestions.append(
                f"Excluded {self.rejected_fiducial} small fiducial candidate(s) below "
                f"{self.fiducial_filter_ratio:.0%} of the largest target area."
            )
        if not suggestions and self.accepted_count:
            suggestions.append("Detection looks healthy.")
        return suggestions


@dataclass
class RawNormalizationReport:
    mode: str
    observed_min: float
    observed_max: float
    black_level: float
    white_level: float
    effective_bit_depth: int | None = None
    alignment: str | None = None


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
    radius = width // 2
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


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
    expected_bytes = header + expected_values * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        packed_hint = ""
        if dtype.itemsize == 2 and actual_bytes < expected_bytes:
            packed_hint = " The stream may use packed 10/12/14-bit encoding and need unpacking before import."
        relation = "ended early" if actual_bytes < expected_bytes else "contains trailing data"
        raise ValueError(
            f"raw input {relation}; expected exactly {expected_bytes} bytes including the {header}-byte header, "
            f"found {actual_bytes}.{packed_hint}"
        )
    with path.open("rb") as fin:
        fin.seek(header)
        data = np.fromfile(fin, dtype=dtype, count=expected_values)
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


def infer_raw_encoding(img: np.ndarray) -> tuple[int | None, str | None]:
    if not np.issubdtype(img.dtype, np.integer):
        return None, None
    storage_bits = np.iinfo(img.dtype).bits
    values = img.astype(np.int64, copy=False).ravel()
    if values.size == 0 or np.issubdtype(img.dtype, np.signedinteger) and np.min(values) < 0:
        return None, None
    observed_max = int(np.max(values))
    candidates = [bits for bits in (8, 10, 12, 14, 16) if bits <= storage_bits]
    right_bits = next((bits for bits in candidates if observed_max <= (1 << bits) - 1), storage_bits)
    if storage_bits > 8:
        for bits in candidates:
            shift = storage_bits - bits
            if shift <= 0:
                continue
            sample = values[:: max(1, values.size // 100000)]
            if np.mean((sample & ((1 << shift) - 1)) == 0) >= 0.98 and observed_max > (1 << bits) - 1:
                return bits, "left"
    return right_bits, "right"


def normalize_raw_image(
    img: np.ndarray,
    mode: str = "auto",
    bit_depth: int = 16,
    alignment: str = "right",
    black_level: float | None = None,
    white_level: float | None = None,
) -> tuple[np.ndarray, RawNormalizationReport]:
    arr = img.astype(np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("raw image does not contain finite values")
    if not finite.all():
        raise ValueError("raw image contains non-finite values")
    observed = arr[finite]
    observed_min = float(np.min(observed))
    observed_max = float(np.max(observed))
    inferred_bits, inferred_alignment = infer_raw_encoding(img)

    if mode == "auto":
        if observed.size < 1000:
            low, high = observed_min, observed_max
        else:
            low, high = (float(value) for value in np.percentile(observed, (0.1, 99.9)))
            if high <= low:
                low, high = observed_min, observed_max
    elif mode == "bit-depth":
        if not np.issubdtype(img.dtype, np.integer):
            raise ValueError("bit-depth normalization requires an integer raw data type")
        storage_bits = np.iinfo(img.dtype).bits
        if bit_depth <= 0 or bit_depth > storage_bits:
            raise ValueError(f"raw bit depth must be between 1 and the {storage_bits}-bit storage width")
        low = 0.0
        high = float(((1 << bit_depth) - 1) << (storage_bits - bit_depth) if alignment == "left" else (1 << bit_depth) - 1)
    elif mode == "manual":
        if black_level is None or white_level is None:
            raise ValueError("manual raw normalization requires black and white levels")
        low, high = float(black_level), float(white_level)
    elif mode == "dtype-range":
        normalized = normalize_image(img)
        info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
        low = float(info.min) if info is not None else observed_min
        high = float(info.max) if info is not None else observed_max
        return normalized, RawNormalizationReport(
            mode, observed_min, observed_max, low, high, inferred_bits, inferred_alignment
        )
    else:
        raise ValueError(f"unsupported raw normalization mode {mode}")

    if mode == "auto" and high <= low:
        return np.zeros_like(arr), RawNormalizationReport(
            mode, observed_min, observed_max, low, high, inferred_bits, inferred_alignment
        )
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError(f"raw normalization white level ({high:g}) must be greater than black level ({low:g})")
    normalized = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return normalized, RawNormalizationReport(
        mode, observed_min, observed_max, low, high, inferred_bits, inferred_alignment
    )


def display_copy(img: np.ndarray) -> np.ndarray:
    arr = normalize_image(img)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def luminance_from_array(
    img: np.ndarray,
    linear: bool,
    invert: bool,
    apply_srgb_for_uint8: bool,
    normalized: bool = False,
    channel_order: str = "bgr",
) -> tuple[np.ndarray, np.ndarray]:
    if img.dtype not in (np.uint8, np.uint16) and not (
        np.issubdtype(img.dtype, np.signedinteger) or np.issubdtype(img.dtype, np.floating)
    ):
        raise ValueError("invalid image type; numeric 8-bit, 16-bit, signed integer, or float images are supported")
    if channel_order not in ("rgb", "bgr"):
        raise ValueError("channel order must be rgb or bgr")

    original = img.copy()
    arr = img.astype(np.float64) if normalized else normalize_image(img)
    if arr.ndim == 2:
        lum = arr
    else:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        bgr = arr if channel_order == "bgr" else arr[:, :, ::-1]
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


def threshold_dark_objects(
    lum: np.ndarray,
    threshold: float,
    threshold_window: float,
    threshold_mode: str = "hybrid",
) -> np.ndarray:
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
    if threshold_mode == "adaptive":
        mask = adaptive
    elif threshold_mode == "global":
        mask = global_mask
    else:
        mask = cv2.bitwise_and(adaptive, global_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def order_box_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def detect_boxes(
    lum: np.ndarray,
    threshold: float,
    threshold_window: float,
    threshold_mode: str = "hybrid",
    min_relative_area: float = 0.0,
) -> list[np.ndarray]:
    boxes, _report = detect_boxes_with_diagnostics(
        lum, threshold, threshold_window, threshold_mode, min_relative_area
    )
    return boxes


def detect_boxes_with_diagnostics(
    lum: np.ndarray,
    threshold: float,
    threshold_window: float,
    threshold_mode: str = "hybrid",
    min_relative_area: float = 0.0,
) -> tuple[list[np.ndarray], DetectionReport]:
    require_cv2()
    if not 0.0 <= min_relative_area <= 1.0:
        raise ValueError("relative fiducial area filter must be in the interval [0, 1]")
    mask = threshold_dark_objects(lum, threshold, threshold_window, threshold_mode)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = lum.shape[0] * lum.shape[1]
    report = DetectionReport(
        threshold_mode,
        threshold,
        threshold_window,
        contour_count=len(contours),
        fiducial_filter_ratio=min_relative_area,
    )
    boxes: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(60.0, image_area * 0.00005):
            report.rejected_small_area += 1
            continue
        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        if min(width, height) < 8:
            report.rejected_short_side += 1
            continue
        rectangularity = area / max(width * height, 1.0)
        if rectangularity < 0.55:
            report.rejected_shape += 1
            LOGGER.warning("skipping low-rectangularity contour with score %.2f", rectangularity)
            continue
        boxes.append(order_box_points(cv2.boxPoints(rect).astype(np.float64)))
    if boxes and min_relative_area > 0.0:
        areas = np.array([float(cv2.contourArea(box.astype(np.float32))) for box in boxes])
        keep = areas >= float(np.max(areas)) * min_relative_area
        report.rejected_fiducial = int(np.count_nonzero(~keep))
        boxes = [box for box, accepted in zip(boxes, keep) if accepted]
    boxes.sort(key=lambda box: (box[:, 1].mean(), box[:, 0].mean()))
    report.accepted_count = len(boxes)
    return boxes, report


def auto_tune_detection(lum: np.ndarray, min_relative_area: float = 0.0) -> tuple[list[np.ndarray], DetectionReport]:
    candidates: list[tuple[float, list[np.ndarray], DetectionReport]] = []
    for mode in ("hybrid", "adaptive", "global"):
        for threshold in (0.4, 0.5, 0.6, 0.7):
            for window in (0.15, 0.25, 1.0 / 3.0, 0.5):
                boxes, report = detect_boxes_with_diagnostics(
                    lum, threshold, window, mode, min_relative_area
                )
                areas = np.array([float(cv2.contourArea(box.astype(np.float32))) for box in boxes])
                if areas.size:
                    representative = areas >= max(float(np.max(areas)) * 0.05, float(np.median(areas)) * 0.2)
                    score_boxes = [box for box, keep in zip(boxes, representative) if keep]
                    score_areas = areas[representative]
                    consistency = float(np.min(score_areas) / np.max(score_areas)) if score_areas.size > 1 else 1.0
                    score = float(np.sum(score_areas)) * (1.0 + math.log1p(len(score_boxes))) * (0.5 + 0.5 * consistency)
                else:
                    score_boxes = []
                    score = 0.0
                score -= report.rejected_shape * 1000.0
                candidates.append((score, score_boxes, report))
    _score, boxes, report = max(candidates, key=lambda item: item[0])
    report.accepted_count = len(boxes)
    return boxes, report


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


def orient_esf(esf: np.ndarray, oversampling: int) -> tuple[np.ndarray, float]:
    low_mean = float(np.mean(esf[:oversampling * 4]))
    high_mean = float(np.mean(esf[-oversampling * 4:]))
    if low_mean > high_mean:
        esf = esf[::-1]
    return esf, abs(high_mean - low_mean)


def refine_edge_line(
    lum: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    search_radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    tangent = p1 - p0
    length = float(np.linalg.norm(tangent))
    if length < 8:
        return p0, p1
    tangent /= length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    center = (p0 + p1) * 0.5
    margin = search_radius + 2.0
    min_x = max(0, int(math.floor(min(p0[0], p1[0]) - margin)))
    max_x = min(lum.shape[1] - 1, int(math.ceil(max(p0[0], p1[0]) + margin)))
    min_y = max(0, int(math.floor(min(p0[1], p1[1]) - margin)))
    max_y = min(lum.shape[0] - 1, int(math.ceil(max(p0[1], p1[1]) + margin)))
    grid_x, grid_y = np.meshgrid(
        np.arange(min_x, max_x + 1, dtype=np.float64),
        np.arange(min_y, max_y + 1, dtype=np.float64),
    )
    delta_x = grid_x - center[0]
    delta_y = grid_y - center[1]
    along = delta_x * tangent[0] + delta_y * tangent[1]
    across = delta_x * normal[0] + delta_y * normal[1]
    selected = (np.abs(along) <= 0.42 * length) & (np.abs(across) <= search_radius)
    if np.count_nonzero(selected) < 24:
        return p0, p1

    local_lum = lum[min_y : max_y + 1, min_x : max_x + 1].astype(np.float64)
    grad_y, grad_x = np.gradient(local_lum)
    weights = np.abs(grad_x * normal[0] + grad_y * normal[1])
    local_weights = weights[selected]
    cutoff = float(np.percentile(local_weights, 70))
    strong = selected & (weights >= max(cutoff, 1e-6))
    if np.count_nonzero(strong) < 12:
        return p0, p1
    points = np.column_stack((grid_x[strong], grid_y[strong]))
    point_weights = weights[strong]
    refined_center = np.average(points, axis=0, weights=point_weights)
    centered = points - refined_center
    covariance = (centered * point_weights[:, None]).T @ centered / np.sum(point_weights)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    refined_tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    if np.dot(refined_tangent, tangent) < 0:
        refined_tangent = -refined_tangent
    angle_error = math.degrees(math.acos(float(np.clip(np.dot(refined_tangent, tangent), -1.0, 1.0))))
    offset = abs(float(np.dot(refined_center - center, normal)))
    if angle_error > 2.0 or offset > search_radius:
        return p0, p1
    return refined_center - refined_tangent * length * 0.5, refined_center + refined_tangent * length * 0.5


def interpolated_esf_from_edge(
    lum: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    oversampling: int = 8,
    radius: float = 12.0,
) -> EsfResult:
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

    esf, contrast = orient_esf(esf, oversampling)
    return EsfResult(esf, 1.0 / oversampling, contrast, "interpolated", 1.0)


def pixel_binned_esf_from_edge(
    lum: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    oversampling: int = 8,
    radius: float = 12.0,
) -> EsfResult:
    tangent = p1 - p0
    length = float(np.linalg.norm(tangent))
    if length < 8:
        raise ValueError("edge is too short")
    tangent /= length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    center = (p0 + p1) * 0.5

    margin = radius + 2.0
    min_x = max(0, int(math.floor(min(p0[0], p1[0]) - margin)))
    max_x = min(lum.shape[1] - 1, int(math.ceil(max(p0[0], p1[0]) + margin)))
    min_y = max(0, int(math.floor(min(p0[1], p1[1]) - margin)))
    max_y = min(lum.shape[0] - 1, int(math.ceil(max(p0[1], p1[1]) + margin)))
    grid_x, grid_y = np.meshgrid(
        np.arange(min_x, max_x + 1, dtype=np.float64),
        np.arange(min_y, max_y + 1, dtype=np.float64),
    )
    delta_x = grid_x - center[0]
    delta_y = grid_y - center[1]
    along = delta_x * tangent[0] + delta_y * tangent[1]
    across = delta_x * normal[0] + delta_y * normal[1]
    selected = (np.abs(along) <= 0.45 * length) & (across >= -radius) & (across <= radius)
    if int(np.count_nonzero(selected)) < oversampling * 8:
        raise ValueError("not enough source pixels for pixel-binned ESF")

    spacing = 1.0 / oversampling
    bin_count = int(round(2.0 * radius * oversampling)) + 1
    bin_indices = np.floor((across[selected] + radius) / spacing).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)
    sums = np.bincount(bin_indices, weights=lum[min_y : max_y + 1, min_x : max_x + 1][selected], minlength=bin_count)
    counts = np.bincount(bin_indices, minlength=bin_count)
    valid = counts > 0
    occupancy = float(np.mean(valid))
    if int(np.count_nonzero(valid)) < 8:
        raise ValueError("pixel-binned ESF has too few populated bins")
    esf = np.empty(bin_count, dtype=np.float64)
    esf[valid] = sums[valid] / counts[valid]
    esf[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), esf[valid])
    esf, contrast = orient_esf(esf, oversampling)
    return EsfResult(esf, spacing, contrast, "pixel-binned", occupancy)


def create_esf_from_edge(
    lum: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    oversampling: int = 8,
    radius: float = 12.0,
    method: str = "pixel-binned",
) -> EsfResult:
    if method == "interpolated":
        return interpolated_esf_from_edge(lum, p0, p1, oversampling, radius)
    if method == "auto":
        try:
            result = pixel_binned_esf_from_edge(lum, p0, p1, oversampling, radius)
            if result.bin_occupancy >= 0.5 and folded_edge_angle(p0, p1) >= 1.0:
                return result
        except ValueError:
            pass
        return interpolated_esf_from_edge(lum, p0, p1, oversampling, radius)
    return pixel_binned_esf_from_edge(lum, p0, p1, oversampling, radius)


def esf_from_edge(
    lum: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    oversampling: int = 8,
    radius: float = 12.0,
    method: str = "pixel-binned",
) -> tuple[np.ndarray, float, float]:
    result = create_esf_from_edge(lum, p0, p1, oversampling, radius, method)
    return result.values, result.sample_spacing, result.contrast


def sfr_from_esf(esf: np.ndarray, sample_spacing: float, smooth: bool, full_sfr: bool) -> tuple[np.ndarray, np.ndarray]:
    esf = smooth_esf(esf.astype(np.float64), smooth)
    lsf = np.gradient(esf, sample_spacing)
    if lsf.size > 1:
        lsf *= np.hamming(lsf.size)
    response = np.abs(np.fft.rfft(lsf))
    if response.size == 0 or not math.isfinite(float(response[0])) or response[0] <= 1e-12:
        raise ValueError("empty SFR response")
    freqs = np.fft.rfftfreq(lsf.size, d=sample_spacing)
    argument = 2.0 * np.pi * freqs * sample_spacing
    derivative_correction = np.ones_like(argument)
    nonzero = np.abs(argument) > 1e-12
    sine = np.sin(argument[nonzero])
    derivative_correction[nonzero] = np.minimum(
        np.abs(np.divide(argument[nonzero], sine, out=np.full_like(sine, 10.0), where=np.abs(sine) > 1e-12)),
        10.0,
    )
    sfr = response / response[0] * derivative_correction
    max_freq = 2.0 if full_sfr else 1.0
    keep = freqs <= max_freq
    return freqs[keep], sfr[keep]


def edge_profiles_from_esf(esf: np.ndarray, sample_spacing: float, smooth: bool) -> tuple[np.ndarray, np.ndarray]:
    smoothed_esf = smooth_esf(esf.astype(np.float64), smooth)
    return smoothed_esf, np.gradient(smoothed_esf, sample_spacing)


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
    roi_radius: float = 12.0,
    esf_method: str = "pixel-binned",
) -> EdgeMeasurement:
    p0, p1 = refine_edge_line(lum, p0, p1)
    esf_result = create_esf_from_edge(lum, p0, p1, radius=roi_radius, method=esf_method)
    esf = esf_result.values
    spacing = esf_result.sample_spacing
    edge_contrast = esf_result.contrast
    freqs, sfr = sfr_from_esf(esf, spacing, smooth=smooth, full_sfr=full_sfr)
    display_esf, display_lsf = edge_profiles_from_esf(esf, spacing, smooth=smooth)
    mtf_value = reported_mtf_value(freqs, sfr, mtf_metric, mtf_contrast, pixel_size)
    center = (p0 + p1) * 0.5
    angle = folded_edge_angle(p0, p1)
    notes: list[str] = []
    if edge_contrast < 0.1:
        notes.append("low contrast")
    if angle < 1.0:
        notes.append("edge angle is nearly axis-aligned")
    if esf_result.method == "pixel-binned" and esf_result.bin_occupancy < 0.5:
        notes.append(f"sparse pixel bins ({esf_result.bin_occupancy:.0%} occupied)")
    if esf_result.method == "interpolated" and esf_method == "auto":
        notes.append("pixel binning unsuitable; used interpolated fallback")
    if not math.isfinite(mtf_value):
        notes.append("reported MTF could not be resolved")
    quality_score = float(np.clip(edge_contrast / 0.35, 0.0, 1.0))
    if angle < 1.0:
        quality_score *= 0.65
    if esf_result.method == "pixel-binned" and esf_result.bin_occupancy < 0.5:
        quality_score *= max(0.4, esf_result.bin_occupancy / 0.5)
    if not math.isfinite(mtf_value):
        quality_score *= 0.4
    quality_label = "Good" if quality_score >= 0.7 else "Review" if quality_score >= 0.4 else "Poor"
    return EdgeMeasurement(
        block_id=block_id,
        edge_x=float(center[0]),
        edge_y=float(center[1]),
        mtf_value=mtf_value,
        mtf_metric=mtf_metric,
        mtf_column=mtf_metric_column(mtf_metric, pixel_size),
        corner_x=float(corner[0]),
        corner_y=float(corner[1]),
        edge_angle=angle,
        radial_angle=radial_angle(center, lum.shape),
        sfr=resample_sfr(freqs, sfr, full_sfr),
        quality=edge_contrast,
        edge_start_x=float(p0[0]),
        edge_start_y=float(p0[1]),
        edge_end_x=float(p1[0]),
        edge_end_y=float(p1[1]),
        esf=display_esf,
        lsf=display_lsf,
        sample_spacing=spacing,
        quality_score=quality_score,
        quality_label=quality_label,
        quality_notes=tuple(notes),
        esf_method=esf_result.method,
        bin_occupancy=esf_result.bin_occupancy,
    )


def load_input_luminance(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
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
        normalized, report = normalize_raw_image(
            raw_img,
            mode=getattr(args, "raw_normalization", "auto"),
            bit_depth=getattr(args, "raw_bit_depth", 16),
            alignment=getattr(args, "raw_alignment", "right"),
            black_level=getattr(args, "raw_black_level", None),
            white_level=getattr(args, "raw_white_level", None),
        )
        args.raw_normalization_report = report
        LOGGER.info(
            "raw samples %.6g..%.6g; normalized %.6g..%.6g using %s; likely %s-bit %s-aligned",
            report.observed_min,
            report.observed_max,
            report.black_level,
            report.white_level,
            report.mode,
            report.effective_bit_depth if report.effective_bit_depth is not None else "unknown",
            report.alignment or "unknown",
        )
        return luminance_from_array(
            normalized,
            linear=True,
            invert=args.invert,
            apply_srgb_for_uint8=False,
            normalized=True,
            channel_order=getattr(args, "raw_channel_order", "rgb"),
        )
    return load_luminance(input_path, linear=args.linear, invert=args.invert)


def box_edges(box: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for idx in range(4):
        yield box[idx], box[(idx + 1) % 4], box[idx]


def detect_single_roi_box(
    lum: np.ndarray,
    threshold: float,
    threshold_window: float = 1.0 / 3.0,
    threshold_mode: str = "hybrid",
) -> np.ndarray:
    mask = threshold_dark_objects(lum, threshold, threshold_window, threshold_mode)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no edge-like object found in single ROI")
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    return order_box_points(cv2.boxPoints(rect).astype(np.float64))


def analyze_image(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[EdgeMeasurement]]:
    lum, original = load_input_luminance(args)
    min_relative_area = getattr(args, "fiducial_max_area_ratio", 0.0) if getattr(
        args, "exclude_small_fiducials", False
    ) else 0.0
    manual_boxes = getattr(args, "manual_boxes", None)
    if manual_boxes is not None:
        boxes = [np.asarray(box, dtype=np.float64) for box in manual_boxes]
    elif getattr(args, "auto_tune", False):
        boxes, report = auto_tune_detection(lum, min_relative_area)
        args.threshold_mode = report.threshold_mode
        args.threshold = report.threshold
        args.threshold_window = report.threshold_window
    else:
        boxes = [detect_single_roi_box(lum, args.threshold, args.threshold_window, args.threshold_mode)] if args.single_roi else detect_boxes(
            lum, args.threshold, args.threshold_window, args.threshold_mode, min_relative_area
        )
    excluded = set(getattr(args, "excluded_blocks", []))
    indexed_boxes = [(idx, box) for idx, box in enumerate(boxes, start=1) if idx not in excluded]
    if not indexed_boxes:
        raise ValueError("no dark rectangular objects found; adjust threshold mode or detection settings")

    measurements: list[EdgeMeasurement] = []
    for block_id, box in indexed_boxes:
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
                    roi_radius=args.roi_radius,
                    esf_method=args.esf_method,
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


def annotation_style(shape: tuple[int, ...]) -> tuple[int, int, int, float, int, int, int, int]:
    scale = float(np.clip(min(shape[:2]) / 800.0, 0.25, 1.7))
    outer_radius = max(2, int(round(6 * scale)))
    middle_radius = max(1, int(round(4 * scale)))
    inner_radius = max(1, int(round(3 * scale)))
    font_scale = float(np.clip(0.48 * scale, 0.14, 0.72))
    white_thickness = max(1, int(round(6 * scale)))
    black_thickness = max(1, int(round(4 * scale)))
    color_thickness = max(1, int(round(2 * scale)))
    offset = max(2, int(round(7 * scale)))
    return (
        outer_radius,
        middle_radius,
        inner_radius,
        font_scale,
        white_thickness,
        black_thickness,
        color_thickness,
        offset,
    )


def make_annotation(
    lum: np.ndarray,
    original: np.ndarray,
    measurements: Sequence[EdgeMeasurement],
    label_mode: str = "All values",
) -> np.ndarray:
    require_cv2()
    if original.ndim == 2:
        annotated = luminance_to_bgr(lum)
    else:
        base = display_copy(original[:, :, :3])
        annotated = base.copy()
    height, width = annotated.shape[:2]
    outer_radius, middle_radius, inner_radius, font_scale, white_thickness, black_thickness, color_thickness, offset = annotation_style(
        annotated.shape
    )
    for m in measurements:
        color = ANNOTATION_COLOR_BGR
        if not math.isfinite(m.mtf_value):
            label = "N/A"
        else:
            label = f"{m.mtf_value:.1f}" if m.mtf_value >= 10 else f"{m.mtf_value:.3f}"
        pos = (int(round(m.edge_x)), int(round(m.edge_y)))
        cv2.circle(annotated, pos, outer_radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, pos, middle_radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(annotated, pos, inner_radius, color, -1, cv2.LINE_AA)
        if label_mode == "Markers only":
            continue
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_thickness
        )
        text_x = min(max(pos[0] + offset, 0), max(width - text_width - 1, 0))
        text_y = min(max(pos[1] - offset, text_height + baseline), max(height - baseline - 1, text_height))
        text_pos = (text_x, text_y)
        cv2.putText(annotated, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), white_thickness, cv2.LINE_AA)
        cv2.putText(annotated, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), black_thickness, cv2.LINE_AA)
        cv2.putText(annotated, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, color_thickness, cv2.LINE_AA)
    return annotated


def make_detection_preview(
    lum: np.ndarray,
    original: np.ndarray,
    boxes: Sequence[np.ndarray],
    excluded_blocks: Sequence[int] = (),
    selected_block: int | None = None,
) -> np.ndarray:
    require_cv2()
    if original.ndim == 2:
        preview = luminance_to_bgr(lum)
    else:
        preview = display_copy(original[:, :, :3]).copy()
    excluded = set(excluded_blocks)
    for block_id, box in enumerate(boxes, start=1):
        color = (120, 120, 120) if block_id in excluded else (0, 210, 255)
        points = np.round(box).astype(np.int32)
        thickness = 6 if block_id == selected_block else 3
        cv2.polylines(preview, [points], True, color, thickness, cv2.LINE_AA)
        center = tuple(np.round(box.mean(axis=0)).astype(int))
        label = f"{block_id} excluded" if block_id in excluded else str(block_id)
        cv2.putText(preview, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return preview


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


def edge_tables_csv(measurements: Sequence[EdgeMeasurement]) -> dict[str, str]:
    mtf_out = io.StringIO()
    mtf_writer = csv.writer(mtf_out)
    metric_column = measurements[0].mtf_column if measurements else "mtf_ny4"
    mtf_writer.writerow(["block_id", "edge_x", "edge_y", metric_column, "corner_x", "corner_y"])
    for m in measurements:
        mtf_writer.writerow(
            [
                m.block_id,
                f"{m.edge_x:.6f}",
                f"{m.edge_y:.6f}",
                f"{m.mtf_value:.9g}",
                f"{m.corner_x:.6f}",
                f"{m.corner_y:.6f}",
            ]
        )

    sfr_out = io.StringIO()
    sfr_writer = csv.writer(sfr_out)
    sfr_count = len(measurements[0].sfr) if measurements else 0
    sfr_writer.writerow(
        ["block_id", "edge_x", "edge_y", "edge_angle", "radial_angle"]
        + [f"sfr_{idx:03d}" for idx in range(sfr_count)]
    )
    for m in measurements:
        sfr_writer.writerow(
            [
                m.block_id,
                f"{m.edge_x:.6f}",
                f"{m.edge_y:.6f}",
                f"{m.edge_angle:.6f}",
                f"{m.radial_angle:.6f}",
            ]
            + [f"{v:.9g}" for v in m.sfr]
        )
    return {"edge_mtf_values.csv": mtf_out.getvalue(), "edge_sfr_values.csv": sfr_out.getvalue()}


def write_annotation(output_dir: Path, annotated: np.ndarray) -> None:
    require_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "annotated.png"
    if not cv2.imwrite(str(out_path), annotated):
        raise ValueError(f"could not write {out_path}")


def write_diagnostics(
    output_dir: Path,
    report: DetectionReport,
    measurements: Sequence[EdgeMeasurement],
    raw_report: RawNormalizationReport | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = diagnostics_payload(report, measurements, raw_report)
    (output_dir / "analysis_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


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

    margin = max(2, min(18, min(width, height) // 8))
    bar_w = max(2, min(28, width - 2 * margin))
    bar_h = max(2, min(height - 2 * margin, 240))
    x0 = max(width - margin - bar_w, 0)
    y0 = min(margin, max(height - bar_h, 0))
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    colorbar = cv2.applyColorMap(np.repeat(gradient, bar_w, axis=1), cv2.COLORMAP_JET)
    output[y0 : y0 + bar_h, x0 : x0 + bar_w] = colorbar
    cv2.rectangle(output, (x0, y0), (min(x0 + bar_w, width - 1), min(y0 + bar_h, height - 1)), (255, 255, 255), 1, cv2.LINE_AA)
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


def summarize_measurements(measurements: Sequence[EdgeMeasurement]) -> dict[str, float | int]:
    if not measurements:
        return {"edges": 0, "blocks": 0, "median": 0.0, "minimum": 0.0, "maximum": 0.0}
    values = np.array([measurement.mtf_value for measurement in measurements], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        median = minimum = maximum = float("nan")
    else:
        median = float(np.median(finite))
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
    return {
        "edges": len(measurements),
        "blocks": len({measurement.block_id for measurement in measurements}),
        "median": median,
        "minimum": minimum,
        "maximum": maximum,
    }


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def diagnostics_payload(
    report: DetectionReport,
    measurements: Sequence[EdgeMeasurement],
    raw_report: RawNormalizationReport | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "detection": {
            **report.__dict__,
            "suggestions": report.suggestions(),
        },
        "edge_quality": [
            {
                "block_id": m.block_id,
                "edge_x": m.edge_x,
                "edge_y": m.edge_y,
                "score": m.quality_score,
                "label": m.quality_label,
                "notes": list(m.quality_notes),
                "esf_method": m.esf_method,
                "bin_occupancy": m.bin_occupancy,
            }
            for m in measurements
        ],
    }
    if raw_report is not None:
        payload["raw_normalization"] = raw_report.__dict__
    return payload


def image_to_data_url(image: np.ndarray, ext: str = ".png") -> str:
    require_cv2()
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("could not encode preview image")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def display_image_from_original(lum: np.ndarray, original: np.ndarray) -> np.ndarray:
    if original.ndim == 2:
        return cv2.cvtColor(display_copy(original), cv2.COLOR_GRAY2BGR)
    return display_copy(original[:, :, :3])


def web_options_to_namespace(input_path: str | Path, options: dict[str, object] | None = None) -> argparse.Namespace:
    values = options or {}
    raw_enabled = bool(values.get("raw", False))
    threshold_mode = THRESHOLD_MODE_ALIASES.get(str(values.get("threshold_mode", "hybrid")), str(values.get("threshold_mode", "hybrid")))
    esf_method = ESF_METHOD_ALIASES.get(str(values.get("esf_method", "pixel-binned")), str(values.get("esf_method", "pixel-binned")))
    raw_normalization = RAW_NORMALIZATION_ALIASES.get(
        str(values.get("raw_normalization", "auto")), str(values.get("raw_normalization", "auto"))
    )
    fiducial_ratio = (
        float(values["fiducial_max_area_percent"]) / 100.0
        if "fiducial_max_area_percent" in values
        else float(values.get("fiducial_max_area_ratio", 0.2) or 0.2)
    )
    args = argparse.Namespace(
        input_image=str(input_path),
        output_dir=".",
        threshold=float(values.get("threshold", 0.55)),
        threshold_mode=threshold_mode,
        threshold_window=float(values.get("threshold_window", 1.0 / 3.0)),
        roi_radius=float(values.get("roi_radius", 12.0)),
        esf_method=esf_method,
        linear=bool(values.get("linear", False)),
        invert=bool(values.get("invert", False)),
        single_roi=bool(values.get("single_roi", False)),
        mtf_metric=str(values.get("mtf_metric", "mtf_ny4")),
        mtf=float(values.get("mtf", 50.0)),
        annotate=True,
        edges=True,
        heatmap=bool(values.get("heatmap", False)),
        full_sfr=bool(values.get("full_sfr", False)),
        nosmoothing=bool(values.get("nosmoothing", False)),
        pixelsize=values.get("pixelsize"),
        raw=raw_enabled,
        raw_width=values.get("raw_width"),
        raw_height=values.get("raw_height"),
        raw_dtype=str(values.get("raw_dtype", "uint16")),
        raw_byte_order=str(values.get("raw_byte_order", "little")),
        raw_header=int(values.get("raw_header", 0) or 0),
        raw_channels=int(values.get("raw_channels", 1) or 1),
        raw_channel_order=str(values.get("raw_channel_order", "rgb")),
        raw_normalization=raw_normalization,
        raw_bit_depth=int(values.get("raw_bit_depth", 16) or 16),
        raw_alignment=str(values.get("raw_alignment", "right")),
        raw_black_level=values.get("raw_black_level"),
        raw_white_level=values.get("raw_white_level"),
        auto_tune=bool(values.get("auto_tune", False)),
        annotation_labels=str(values.get("annotation_labels", "All values")),
        exclude_small_fiducials=bool(values.get("exclude_small_fiducials", False)),
        fiducial_max_area_ratio=fiducial_ratio,
        manual_boxes=values.get("manual_boxes"),
        excluded_blocks=values.get("excluded_blocks", []),
    )
    if args.pixelsize in ("", None):
        args.pixelsize = None
    else:
        args.pixelsize = float(args.pixelsize)
    if raw_enabled:
        if args.raw_width in ("", None) or args.raw_height in ("", None):
            raise ValueError("raw import requires width and height")
        args.raw_width = int(args.raw_width)
        args.raw_height = int(args.raw_height)
        args.raw_black_level = None if args.raw_black_level in ("", None) else float(args.raw_black_level)
        args.raw_white_level = None if args.raw_white_level in ("", None) else float(args.raw_white_level)
    return args


def web_load_original(input_path: str, options: dict[str, object] | None = None) -> dict[str, object]:
    args = web_options_to_namespace(input_path, options)
    lum, original = load_input_luminance(args)
    image = display_image_from_original(lum, original)
    return {
        "image": image_to_data_url(image),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "raw_normalization": getattr(args, "raw_normalization_report", None).__dict__
        if hasattr(args, "raw_normalization_report")
        else None,
    }


def detect_for_args(args: argparse.Namespace, lum: np.ndarray) -> tuple[list[np.ndarray], DetectionReport]:
    min_relative_area = args.fiducial_max_area_ratio if getattr(args, "exclude_small_fiducials", False) else 0.0
    manual_boxes = getattr(args, "manual_boxes", None)
    if manual_boxes is not None:
        boxes = [np.asarray(box, dtype=np.float64) for box in manual_boxes]
        report = DetectionReport(args.threshold_mode, args.threshold, args.threshold_window, accepted_count=len(boxes))
    elif getattr(args, "auto_tune", False):
        boxes, report = auto_tune_detection(lum, min_relative_area)
        args.threshold_mode = report.threshold_mode
        args.threshold = report.threshold
        args.threshold_window = report.threshold_window
    elif getattr(args, "single_roi", False):
        boxes = [detect_single_roi_box(lum, args.threshold, args.threshold_window, args.threshold_mode)]
        report = DetectionReport(args.threshold_mode, args.threshold, args.threshold_window, accepted_count=1)
    else:
        boxes, report = detect_boxes_with_diagnostics(
            lum, args.threshold, args.threshold_window, args.threshold_mode, min_relative_area
        )
    return boxes, report


def web_preview_detection(input_path: str, options: dict[str, object] | None = None) -> dict[str, object]:
    args = web_options_to_namespace(input_path, options)
    lum, original = load_input_luminance(args)
    boxes, report = detect_for_args(args, lum)
    preview = make_detection_preview(lum, original, boxes, getattr(args, "excluded_blocks", []))
    mask = threshold_dark_objects(lum, report.threshold, report.threshold_window, report.threshold_mode)
    return {
        "detection_image": image_to_data_url(preview),
        "threshold_image": image_to_data_url(mask),
        "boxes": [box.tolist() for box in boxes],
        "report": {**report.__dict__, "suggestions": report.suggestions()},
        "raw_normalization": getattr(args, "raw_normalization_report", None).__dict__
        if hasattr(args, "raw_normalization_report")
        else None,
    }


def measurement_to_dict(measurement: EdgeMeasurement) -> dict[str, object]:
    return {
        "block_id": measurement.block_id,
        "edge_x": measurement.edge_x,
        "edge_y": measurement.edge_y,
        "mtf_value": measurement.mtf_value,
        "mtf_metric": measurement.mtf_metric,
        "mtf_column": measurement.mtf_column,
        "corner_x": measurement.corner_x,
        "corner_y": measurement.corner_y,
        "edge_angle": measurement.edge_angle,
        "radial_angle": measurement.radial_angle,
        "quality": measurement.quality,
        "edge_start_x": measurement.edge_start_x,
        "edge_start_y": measurement.edge_start_y,
        "edge_end_x": measurement.edge_end_x,
        "edge_end_y": measurement.edge_end_y,
        "sample_spacing": measurement.sample_spacing,
        "quality_score": measurement.quality_score,
        "quality_label": measurement.quality_label,
        "quality_notes": list(measurement.quality_notes),
        "esf_method": measurement.esf_method,
        "bin_occupancy": measurement.bin_occupancy,
        "sfr": measurement.sfr.tolist(),
        "esf": measurement.esf.tolist(),
        "lsf": measurement.lsf.tolist(),
    }


def web_analyze(input_path: str, options: dict[str, object] | None = None) -> dict[str, object]:
    args = web_options_to_namespace(input_path, options)
    lum_for_detection, _original_for_detection = load_input_luminance(args)
    _boxes, report = detect_for_args(args, lum_for_detection)
    lum, annotated, measurements = analyze_image(args)
    outputs = {
        "annotated.png": image_to_data_url(annotated),
        "analysis_diagnostics.json": json.dumps(
            diagnostics_payload(report, measurements, getattr(args, "raw_normalization_report", None)),
            indent=2,
        ),
        **edge_tables_csv(measurements),
    }
    if getattr(args, "heatmap", False):
        outputs["mtf_heatmap.png"] = image_to_data_url(make_mtf_heatmap(lum, measurements))
    return {
        "summary": summarize_measurements(measurements),
        "measurements": [measurement_to_dict(measurement) for measurement in measurements],
        "outputs": outputs,
        "report": {**report.__dict__, "suggestions": report.suggestions()},
        "raw_normalization": getattr(args, "raw_normalization_report", None).__dict__
        if hasattr(args, "raw_normalization_report")
        else None,
    }


def prepare_output_dir(output_dir: Path, annotate: bool, edges: bool, heatmap: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled = {
        "annotated.png": annotate,
        "edge_mtf_values.csv": edges,
        "edge_sfr_values.csv": edges,
        "mtf_heatmap.png": heatmap,
    }
    for filename, keep in enabled.items():
        path = output_dir / filename
        if not keep and path.exists():
            path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Core Python MTF Mapper analyzer")
    parser.add_argument("input_image", help="input image file name")
    parser.add_argument("output_dir", help="directory for output files")
    parser.add_argument("-t", "--threshold", type=float, default=0.55, help="dark object threshold in [0,1]")
    parser.add_argument(
        "--threshold-mode",
        default="hybrid",
        choices=["hybrid", "adaptive", "global"],
        help="target detection thresholding: adaptive plus global, adaptive only, or global only",
    )
    parser.add_argument("--threshold-window", type=float, default=1.0 / 3.0, help="adaptive threshold window fraction")
    parser.add_argument("--roi-radius", type=float, default=12.0, help="edge sampling radius in pixels")
    parser.add_argument(
        "--esf-method",
        default="pixel-binned",
        choices=["pixel-binned", "interpolated", "auto"],
        help="ESF construction: original-pixel binning, interpolated profiles, or automatic fallback",
    )
    parser.add_argument("--auto-tune", action="store_true", help="search threshold modes and values for the strongest detection")
    parser.add_argument(
        "--exclude-small-fiducials",
        action="store_true",
        help="exclude detected rectangles much smaller than the largest target",
    )
    parser.add_argument(
        "--fiducial-max-area-ratio",
        type=float,
        default=0.2,
        help="exclude candidates below this fraction of the largest target area",
    )
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
    parser.add_argument(
        "--raw-channel-order",
        default="rgb",
        choices=["rgb", "bgr"],
        help="channel order for interleaved color raw streams",
    )
    parser.add_argument(
        "--raw-normalization",
        default="auto",
        choices=["auto", "bit-depth", "manual", "dtype-range"],
        help="map raw samples using robust automatic levels, effective bit depth, manual levels, or the full data type range",
    )
    parser.add_argument("--raw-bit-depth", type=int, default=16, choices=[8, 10, 12, 14, 16], help="effective raw bit depth")
    parser.add_argument("--raw-alignment", default="right", choices=["right", "left"], help="alignment of samples within the storage type")
    parser.add_argument("--raw-black-level", type=float, help="manual raw black level")
    parser.add_argument("--raw-white-level", type=float, help="manual raw white level")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be in the open interval (0, 1)")
    if not 0.0 < args.threshold_window <= 1.0:
        parser.error("--threshold-window must be in the interval (0, 1]")
    if args.roi_radius < 4.0:
        parser.error("--roi-radius must be at least 4 pixels")
    if not 0.0 < args.fiducial_max_area_ratio <= 1.0:
        parser.error("--fiducial-max-area-ratio must be in the interval (0, 1]")
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
        if args.raw_normalization == "manual" and (args.raw_black_level is None or args.raw_white_level is None):
            parser.error("--raw-normalization manual requires --raw-black-level and --raw-white-level")
        if (
            args.raw_normalization == "manual"
            and args.raw_black_level is not None
            and args.raw_white_level is not None
            and args.raw_white_level <= args.raw_black_level
        ):
            parser.error("--raw-white-level must be greater than --raw-black-level")
    elif (
        any(value is not None for value in (args.raw_width, args.raw_height, args.raw_black_level, args.raw_white_level))
        or args.raw_header != 0
        or args.raw_channels != 1
        or args.raw_channel_order != "rgb"
        or args.raw_normalization != "auto"
        or args.raw_bit_depth != 16
        or args.raw_alignment != "right"
    ):
        parser.error("raw metadata options require --raw")
    if not args.annotate and not args.edges and not args.heatmap:
        args.annotate = True
        args.edges = True
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    try:
        lum_for_detection, _original = load_input_luminance(args)
        min_relative_area = args.fiducial_max_area_ratio if args.exclude_small_fiducials else 0.0
        if args.auto_tune:
            _boxes, report = auto_tune_detection(lum_for_detection, min_relative_area)
        elif args.single_roi:
            report = DetectionReport(args.threshold_mode, args.threshold, args.threshold_window, accepted_count=1)
        else:
            _boxes, report = detect_boxes_with_diagnostics(
                lum_for_detection, args.threshold, args.threshold_window, args.threshold_mode, min_relative_area
            )
        lum, annotated, measurements = analyze_image(args)
        output_dir = Path(args.output_dir)
        prepare_output_dir(output_dir, args.annotate, args.edges, args.heatmap)
        if args.edges:
            write_edge_tables(output_dir, measurements)
        if args.annotate:
            write_annotation(output_dir, annotated)
        if args.heatmap:
            write_heatmap(output_dir, lum, measurements)
        write_diagnostics(output_dir, report, measurements, getattr(args, "raw_normalization_report", None))
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("measured %d edges", len(measurements))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
