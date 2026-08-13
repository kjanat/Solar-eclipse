import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def crossing(profile: np.ndarray, level: float) -> float | None:
    indexes = np.where((profile[:-1] < level) & (profile[1:] >= level))[0]
    if not len(indexes):
        return None
    index = min(indexes, key=lambda item: abs(int(item) - len(profile) // 2))
    difference = float(profile[index + 1] - profile[index])
    return (
        float(index + (level - profile[index]) / difference)
        if abs(difference) > 1e-6
        else float(index)
    )


def measure_edge(path: Path) -> tuple[float, int]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    smooth = cv2.GaussianBlur(luminance, (0, 0), 0.6)
    high = float(np.percentile(smooth, 99.7))
    background = float(np.percentile(smooth, 50))
    if high - background < 0.1:
        return math.nan, 0
    normalized = np.clip((smooth - background) / (high - background), 0, 1)
    contours, _ = cv2.findContours(
        (normalized > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    candidates = [contour for contour in contours if cv2.contourArea(contour) > 30]
    if not candidates:
        return math.nan, 0
    contour = max(candidates, key=cv2.contourArea)[:, 0, :]
    gradient_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    widths: list[float] = []
    for x, y in contour[::3]:
        vector_x = float(gradient_x[y, x])
        vector_y = float(gradient_y[y, x])
        norm = math.hypot(vector_x, vector_y)
        if norm < 0.03:
            continue
        offsets = np.linspace(-12, 12, 97, dtype=np.float32)
        xs = np.asarray(x + vector_x / norm * offsets, dtype=np.float32)
        ys = np.asarray(y + vector_y / norm * offsets, dtype=np.float32)
        profile = cv2.remap(
            normalized.astype(np.float32, copy=False),
            xs.reshape(1, -1),
            ys.reshape(1, -1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).ravel()
        if profile[-8:].mean() < profile[:8].mean():
            profile = profile[::-1]
        low = float(np.median(profile[:8]))
        upper = float(np.median(profile[-8:]))
        contrast = upper - low
        if contrast < 0.35:
            continue
        profile = np.convolve((profile - low) / contrast, np.ones(5) / 5, mode="same")

        start = crossing(profile, 0.1)
        end = crossing(profile, 0.9)
        if start is None or end is None or end <= start:
            continue
        width = (end - start) * 0.25
        if 0.3 <= width <= 15:
            widths.append(width)
    return (float(np.median(widths)), len(widths)) if widths else (math.nan, 0)


def calculate_edge_spread(
    metrics_path: Path, crops_dir: Path, output: Path
) -> list[dict[str, str]]:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}; run analyze first")
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Metrics file is empty: {metrics_path}")
    for row in rows:
        crop = crops_dir / f"{Path(row['file']).stem}.jpg"
        if not crop.is_file():
            raise FileNotFoundError(f"Analysis crop not found: {crop}")
        width, samples = measure_edge(crop)
        row["edge_10_90_px"] = str(width)
        row["edge_samples"] = str(samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_edge_rankings(rows: list[dict[str, str]]) -> None:
    print("RANK 10-90 edge width (lower = sharper)")
    for row in sorted(
        rows,
        key=lambda item: (
            float(item["edge_10_90_px"])
            if not math.isnan(float(item["edge_10_90_px"]))
            else 999
        ),
    ):
        print(
            f"{row['file']:12} {float(row['edge_10_90_px']):5.2f}px  "
            f"n={int(row['edge_samples']):4d}  f/{float(row['aperture']):g} "
            f"{float(row['shutter_s']):.6g}s  clip={float(row['clip_all_pct']):.3f}%"
        )
