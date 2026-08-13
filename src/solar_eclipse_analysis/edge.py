import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from solar_eclipse_analysis.models import DataRow

FloatImage = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
IntImage = np.ndarray[tuple[int, int], np.dtype[np.int32]]


def crossing(profile: FloatImage, level: float) -> float | None:
    indexes = np.where((profile[:-1] < level) & (profile[1:] >= level))[0]
    if not len(indexes):
        return None
    index = min(indexes, key=lambda item: abs(int(item) - len(profile) // 2))
    difference = profile.item(index + 1) - profile.item(index)
    return (
        index + (level - profile.item(index)) / difference
        if abs(difference) > 1e-6
        else float(index)
    )


def measure_edge(path: Path) -> tuple[float, int]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    smooth: FloatImage = np.asarray(
        cv2.GaussianBlur(luminance, (0, 0), 0.6), dtype=np.float32
    )
    high = float(np.percentile(smooth, 99.7))
    background = float(np.percentile(smooth, 50))
    if high - background < 0.1:
        return math.nan, 0
    normalized: FloatImage = np.asarray(
        np.clip((smooth - background) / (high - background), 0, 1),
        dtype=np.float32,
    )
    contours, _ = cv2.findContours(
        (normalized > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    candidates = [contour for contour in contours if cv2.contourArea(contour) > 30]
    if not candidates:
        return math.nan, 0
    contour: IntImage = np.asarray(
        max(candidates, key=cv2.contourArea)[:, 0, :], dtype=np.int32
    )
    gradient_x: FloatImage = np.asarray(
        cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3), dtype=np.float32
    )
    gradient_y: FloatImage = np.asarray(
        cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3), dtype=np.float32
    )
    widths: list[float] = []
    for point_index in range(0, len(contour), 3):
        x = contour.item(point_index, 0)
        y = contour.item(point_index, 1)
        vector_x = gradient_x.item(y, x)
        vector_y = gradient_y.item(y, x)
        norm = math.hypot(vector_x, vector_y)
        if norm < 0.03:
            continue
        offsets = np.linspace(-12, 12, 97, dtype=np.float32)
        xs: FloatImage = np.asarray(x + vector_x / norm * offsets, dtype=np.float32)
        ys: FloatImage = np.asarray(y + vector_y / norm * offsets, dtype=np.float32)
        profile: FloatImage = np.asarray(
            cv2.remap(
                normalized,
                xs.reshape(1, -1),
                ys.reshape(1, -1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ).ravel(),
            dtype=np.float32,
        )
        if profile[-8:].mean() < profile[:8].mean():
            profile = profile[::-1]
        low = np.median(profile[:8]).item()
        upper = np.median(profile[-8:]).item()
        contrast = upper - low
        if contrast < 0.35:
            continue
        profile = np.asarray(
            np.convolve((profile - low) / contrast, np.ones(5) / 5, mode="same"),
            dtype=np.float32,
        )

        start = crossing(profile, 0.1)
        end = crossing(profile, 0.9)
        if start is None or end is None or end <= start:
            continue
        width = (end - start) * 0.25
        if 0.3 <= width <= 15:
            widths.append(width)
    return (float(np.median(widths)), len(widths)) if widths else (math.nan, 0)


def calculate_edge_spread(
    metrics_rows: list[DataRow], crops_dir: Path
) -> list[DataRow]:
    rows = [dict(row) for row in metrics_rows]
    if not rows:
        raise RuntimeError("No analysis metrics available")
    for row in rows:
        crop = crops_dir / f"{Path(str(row['file'])).stem}.jpg"
        if not crop.is_file():
            raise FileNotFoundError(f"Analysis crop not found: {crop}")
        width, samples = measure_edge(crop)
        row["edge_10_90_px"] = width
        row["edge_samples"] = samples
    return rows


def print_edge_rankings(rows: list[DataRow]) -> None:
    print("RANK 10-90 edge width (lower = sharper)")
    for row in sorted(
        rows,
        key=lambda item: (
            float(item["edge_10_90_px"])
            if not math.isnan(float(item["edge_10_90_px"]))
            else 999
        ),
    ):
        summary = (
            f"{row['file']:12} {float(row['edge_10_90_px']):5.2f}px  "
            f"n={int(row['edge_samples']):4d}  "
            f"f/{float(row['aperture']):g} "
            f"{float(row['shutter_s']):.6g}s  "
            f"clip={float(row['clip_all_pct']):.3f}%"
        )
        print(summary)
