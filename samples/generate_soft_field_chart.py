#!/usr/bin/env python3
"""Generate a soft synthetic MTF chart whose measured MTF values stay below 1."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


OUT_PATH = Path(__file__).resolve().parent / "mtf_soft_field_chart.png"


def add_soft_rect(
    image: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    angle_deg: float,
    sigma_radial: float,
    sigma_tangential: float,
    radial_angle_deg: float,
) -> None:
    width, height = size
    pad = int(max(56, round(10 * max(sigma_radial, sigma_tangential))))
    patch_w = width + 2 * pad
    patch_h = height + 2 * pad
    mask = np.zeros((patch_h, patch_w), dtype=np.float32)
    rect = ((patch_w / 2.0, patch_h / 2.0), (width, height), angle_deg)
    points = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, points, 1.0)

    matrix = cv2.getRotationMatrix2D((patch_w / 2.0, patch_h / 2.0), radial_angle_deg, 1.0)
    aligned = cv2.warpAffine(mask, matrix, (patch_w, patch_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    aligned = cv2.GaussianBlur(aligned, (0, 0), sigmaX=sigma_radial, sigmaY=sigma_tangential)
    inverse = cv2.invertAffineTransform(matrix)
    mask = cv2.warpAffine(aligned, inverse, (patch_w, patch_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.9, sigmaY=0.9)
    mask = np.clip(mask, 0.0, 1.0)

    cx, cy = center
    x0 = cx - patch_w // 2
    y0 = cy - patch_h // 2
    x1 = x0 + patch_w
    y1 = y0 + patch_h
    ix0 = max(x0, 0)
    iy0 = max(y0, 0)
    ix1 = min(x1, image.shape[1])
    iy1 = min(y1, image.shape[0])
    if ix0 >= ix1 or iy0 >= iy1:
        return
    mx0 = ix0 - x0
    my0 = iy0 - y0
    mx1 = mx0 + (ix1 - ix0)
    my1 = my0 + (iy1 - iy0)
    local_mask = mask[my0:my1, mx0:mx1]
    dark = 38.0
    image[iy0:iy1, ix0:ix1] = image[iy0:iy1, ix0:ix1] * (1.0 - local_mask) + dark * local_mask


def main() -> int:
    width, height = 1600, 1200
    image = np.full((height, width), 232.0, dtype=np.float32)
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    max_radius = float(np.linalg.norm(center))

    xs = np.linspace(220, width - 220, 6)
    ys = np.linspace(180, height - 180, 5)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            point = np.array([x, y], dtype=np.float64)
            delta = point - center
            radius = float(np.linalg.norm(delta) / max_radius)
            radial_angle = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) if radius > 0 else 0.0
            angle = (-7.0 if (row + col) % 2 == 0 else 7.0) + 0.5 * (col - 2.5)

            sigma_radial = 1.6 + 1.9 * radius**1.5
            sigma_tangential = 2.2 + 3.2 * radius**1.7
            add_soft_rect(
                image,
                center=(int(round(x)), int(round(y))),
                size=(118, 82),
                angle_deg=angle,
                sigma_radial=sigma_radial,
                sigma_tangential=sigma_tangential,
                radial_angle_deg=radial_angle,
            )

    yy, xx = np.indices((height, width), dtype=np.float32)
    rr = np.sqrt(((xx - center[0]) / center[0]) ** 2 + ((yy - center[1]) / center[1]) ** 2)
    image -= 8.0 * np.clip(rr, 0.0, 1.0) ** 1.4
    image += 0.9 * np.sin(xx / 100.0) + 0.7 * np.cos(yy / 90.0)
    rng = np.random.default_rng(20260602)
    image += rng.normal(0.0, 0.35, image.shape)

    image = cv2.GaussianBlur(image, (0, 0), sigmaX=0.35, sigmaY=0.35)
    image8 = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT_PATH), image8)
    print(OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
